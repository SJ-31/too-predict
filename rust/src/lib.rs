use core::f64;

use itertools::Itertools;
use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Zip};
use ndarray_stats::CorrelationExt;
use num_traits::zero;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::types::PyModuleMethods;
use pyo3::wrap_pyfunction;
use pyo3::{pyfunction, pymodule, types::PyModule, Bound, PyResult, Python};
use rayon::prelude::*;

// * Send to python

#[pymodule]
fn _rust_helpers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(phi_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(rho_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(encode_pairs, m)?)?;
    Ok(())
}

#[pyfunction]
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
) -> Bound<'py, PyArray2<i64>> {
    let converted: ArrayView2<f64> = arr.as_array();
    let result = encode_pairs_rs(converted);
    result.into_pyarray_bound(py)
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

fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

fn choose(n: u64, r: u64) -> u64 {
    factorial(n) / (factorial(r) * factorial(n - r))
}

/// Encode gene expression pairs as binary indicators, adapted from RaMBat [1]
///
/// # Arguments
/// * arr : An array of shape [ n_samples, n_features ]
///
/// # Returns
/// A binary array of shape [ n_samples, choose(n_features, 2) ]
///
fn encode_pairs_rs(arr: ArrayView2<f64>) -> Array2<i64> {
    let ncols = arr.nrows();
    let n_features = arr.ncols();
    let mut result: Array2<i64> = Array2::zeros([ncols, choose(n_features as u64, 2) as usize]);
    for (i, pair) in (0..n_features).combinations(2).enumerate() {
        let f1: ArrayView1<f64> = arr.slice(s![.., pair[0]]);
        let f2: ArrayView1<f64> = arr.slice(s![.., pair[1]]);
        let comparison = Zip::from(f1)
            .and(f2)
            .map_collect(|x, y| if x > y { 1 } else { 0 });
        result.slice_mut(s![.., i]).assign(&comparison);
    }
    result
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
    let encode2 = encode_pairs_rs(h.view());
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

#[ignore]
#[test]
fn test_combos() {
    let myvec = Vec::from_iter(0..10);
    for (i, b) in myvec.into_iter().combinations(2).enumerate() {
        println!("{:?}, {:?}", i, b[0]);
    }
}
