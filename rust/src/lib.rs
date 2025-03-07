use core::f64;

use itertools::Itertools;
use ndarray::{
    concatenate, s, stack, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Zip,
};
use ndarray_stats::CorrelationExt;
use numpy::{IntoPyArray, PyArray, PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyModuleMethods, PyTuple, PyType};
use pyo3::{pyfunction, pymodule, types::PyModule, Bound, PyResult, Python};
use pyo3::{wrap_pyfunction, IntoPy, PyAny, ToPyObject};
use rayon::prelude::*;

// * References
// [1] Sun, Mengtao, Jieqiong Wang, and Shibiao Wan. “Accurate Identification of Medulloblastoma Subtypes from Diverse Data Sources with Severe Batch Effects by RaMBat.” bioRxiv, February 28, 2025. https://doi.org/10.1101/2025.02.24.640010.
//
//

// * Send to python

#[pymodule]
fn _rust_helpers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(phi_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(rho_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(encode_pairs, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (arr, do_parallel, /))]
fn phi_matrix<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, f64>,
    do_parallel: bool,
) -> Bound<'py, PyArray2<f64>> {
    let converted: ArrayView2<f64> = arr.as_array();
    let result = phi_proportionality_rs(converted, do_parallel);
    result.into_pyarray_bound(py)
}

#[pyfunction]
fn encode_pairs<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Vec<Vec<usize>>)> {
    let math = PyModule::import_bound(py, "math")?;
    let converted: ArrayView2<f64> = arr.as_array();
    let n_features = converted.ncols();
    let comb = math.getattr("comb")?;
    let n: i64 = comb.call1((n_features, 2))?.extract()?;
    println!("{:?}", n);
    let (encoded, names) = encode_pairs_rs(converted);
    let encoded_py = encoded.to_pyarray_bound(py);
    let names_py = names;
    Ok((encoded_py, names_py))
}

#[pyfunction]
fn rho_matrix<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, f64>,
    do_parallel: bool,
) -> Bound<'py, PyArray2<f64>> {
    let converted: ArrayView2<f64> = arr.as_array();
    let result = proportionality_coeff_rs(converted, do_parallel);
    result.into_pyarray_bound(py)
}

// * Rust implementations

fn phi_proportionality_test(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    let log = x.ln();
    (&log - y.ln()).var(1.) / log.var(1.)
}

/// Construct a pairwise matrix of the proportionality coefficient (rho)
///
/// rho has range of -1 to 1
///
/// * arr : a sample x feature matrix
fn proportionality_coeff_rs(arr: ArrayView2<f64>, do_parallel: bool) -> Array2<f64> {
    let cov: Array2<f64> = arr.t().cov(1.0).unwrap();
    let vars: Vec<f64> = arr.var_axis(Axis(0), 1.).to_vec();
    let ncols = arr.ncols();
    let mut result: Array2<f64> = Array2::zeros([ncols, ncols]);
    if !do_parallel {
        for i in 0..ncols {
            let mut calculated_row: Array1<f64> = Array1::zeros(ncols);
            for j in 0..ncols {
                let numer: f64 = 2. * cov[[i, j]];
                let denom = vars.get(i).unwrap() + vars.get(j).unwrap();
                let rho: f64 = numer / denom;
                calculated_row.slice_mut(s![j]).fill(rho);
            }
            result.slice_mut(s![.., i]).assign(&calculated_row);
        }
    } else {
        for i in 0..ncols {
            let tmp: Vec<f64> = (0..ncols)
                .into_par_iter()
                .map(|j| {
                    let numer: f64 = 2. * cov[[i, j]];
                    let denom = vars.get(i).unwrap() + vars.get(j).unwrap();
                    numer / denom
                })
                .collect();
            let row: Array1<f64> = ArrayBase::from_vec(tmp);
            result.slice_mut(s![.., i]).assign(&row);
        }
    }
    result
}

/// Construct a pairwise matrix of Goodness of fit to proportionalty
///
/// * arr : a sample x feature matrix
fn phi_proportionality_rs(arr: ArrayView2<f64>, do_parallel: bool) -> Array2<f64> {
    println!("Computing phi matrix...");
    let ncols = arr.ncols();
    let mut result: Array2<f64> = Array2::zeros([ncols, ncols]);
    let lns: Array2<f64> = arr.ln();
    let vars: Vec<f64> = lns.var_axis(Axis(0), 1.).to_vec();
    if !do_parallel {
        for i in 0..ncols {
            let mut calculated_row: Array1<f64> = Array1::zeros(ncols);
            for j in 0..ncols {
                let x: ArrayView1<f64> = lns.slice(s![.., i]);
                let y: ArrayView1<f64> = lns.slice(s![.., j]);
                let val: f64 = (&x - &y).var(1.) / vars.get(i).unwrap();
                calculated_row.slice_mut(s![j]).fill(val);
            }
            result.slice_mut(s![.., i]).assign(&calculated_row);
        }
    } else {
        for i in 0..ncols {
            let tmp: Vec<f64> = (0..ncols)
                .into_par_iter()
                .map(|j| {
                    let x: ArrayView1<f64> = lns.slice(s![.., i]);
                    let y: ArrayView1<f64> = lns.slice(s![.., j]);
                    let val: f64 = (&x - &y).var(1.) - vars.get(j).unwrap();
                    val
                })
                .collect();
            let row: Array1<f64> = ArrayBase::from_vec(tmp);
            result.slice_mut(s![.., i]).assign(&row);
        }
    }
    result
}

/// Encode gene expression pairs as binary indicators, adapted from RaMBat [1]
///
/// # Arguments
/// * arr : An array of shape [ n_samples, n_features ]
///
/// # Returns
/// A binary array of shape [ n_samples, choose(n_features, 2) ]
///
fn encode_pairs_rs(arr: ArrayView2<f64>) -> (Array2<i64>, Vec<Vec<usize>>) {
    let nrows = arr.nrows();
    let n_features = arr.ncols();
    let mut result: Array2<i64> = Array2::zeros([nrows, 1]);
    let combos: Vec<Vec<usize>> = (0..n_features).combinations(2).collect();
    for pair in (&combos).into_iter() {
        let f1: ArrayView1<f64> = arr.slice(s![.., pair[0]]);
        let f2: ArrayView1<f64> = arr.slice(s![.., pair[1]]);
        let comparison = Zip::from(f1)
            .and(f2)
            .map_collect(|x, y| if x > y { 1 } else { 0 });
        result = concatenate(
            Axis(1),
            &[
                result.view(),
                comparison.to_shape((nrows, 1)).unwrap().view(),
            ],
        )
        .unwrap();
    }
    (result, combos)
}

// * Tests

#[test]
fn test_phi_prop() {
    use ndarray::arr2;
    let h = arr2(&[[8, 1, 2, 3], [4, 5, 6, 7]]).mapv(|x| x as f64);
    let dim = h.raw_dim();
    let s1 = h.slice(s![0, ..]);
    let s2 = h.slice(s![1, ..]);
    let comparison = Zip::from(s1)
        .and_broadcast(s2)
        .map_collect(|x, y| if x > y { 1 } else { 0 });
    let phi2 = phi_proportionality_rs(h.view(), false);
    let (encode2, val) = encode_pairs_rs(h.view());
    println!("foo");
    println!("{:?}", encode2);
}

#[ignore]
#[test]
fn test_prop_coeff() {
    use ndarray::arr2;
    let h = arr2(&[[8, 1, 2, 3], [4, 5, 6, 7]]).mapv(|x| x as f64);
    let rho = proportionality_coeff_rs(h.view(), false);
    println!("{:?}", rho);
}

#[test]
fn test_combos() {
    let myvec = Vec::from_iter(0..15);
    let vs: Vec<Vec<usize>> = myvec.into_iter().combinations(2).collect();
    println!("{:?}", comb_pair_at(5, 8));
}

/// Return the pair at index `query` in a hypothetical sequence of pairs
/// The pair sequence is an ordered sequence of [(0, 1), (0, 2), ..., (j - 2, j - 1)]
///
/// # Arguments
/// * j : the number of elements in the original
/// * query : index of pair sequnce of interest
///
fn comb_pair_at(j: i64, query: i64) -> (i64, i64) {
    let first_cutoffs = (0..j - 1).scan(0, |state, x| {
        *state = *state + (j - 1 - x);
        Some(*state)
    });
    let mut first: i64 = -1;
    let mut f_offset = 0;
    let mut previous = 0;
    for (index, acc) in first_cutoffs.enumerate() {
        if query < acc {
            first = index as i64;
            f_offset = previous;
            break;
        }
        if index > 0 {
            previous = acc;
        }
    }
    if first < 0 {
        panic!("The query is too large!");
    }
    let second = query - f_offset + first + 1;
    (first, second)
}
