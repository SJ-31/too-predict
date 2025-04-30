use core::f64;
use std::cmp::Eq;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use itertools::Itertools;
use ndarray::{
    concatenate, s, stack, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Zip,
};
use ndarray_stats::CorrelationExt;
use numpy::{IntoPyArray, PyArray, PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModuleMethods, PyTuple, PyType};
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
    m.add_function(wrap_pyfunction!(pairwise_overlaps, m)?)?;
    Ok(())
}

#[pyfunction]
fn pairwise_overlaps<'py>(py: Python<'py>, sets: Py<PyDict>, do_parallel: bool) -> Py<PyAny> {
    let converted: HashMap<String, HashSet<String>> = sets.extract(py).unwrap();
    pairwise_overlap_rs(&converted, do_parallel).into_py(py)
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

fn overlap_helper<K, V>(query: &Vec<&K>, dict: &HashMap<K, HashSet<V>>) -> ((K, K), usize)
where
    K: Hash + Eq + Clone,
    V: Hash + Eq,
{
    let x: &K = query[0];
    let y: &K = query[1];
    let xs: &HashSet<V> = dict.get(&x).unwrap();
    let ys: &HashSet<V> = dict.get(&y).unwrap();
    let inter: HashSet<&V> = xs.intersection(ys).collect();
    ((x.clone(), y.clone()), inter.len())
}

fn pairwise_overlap_rs<K, V>(
    sets: &HashMap<K, HashSet<V>>,
    do_parallel: bool,
) -> Vec<((K, K), usize)>
where
    K: Hash + Eq + Send + Sync + Clone,
    V: Hash + Eq + Sync,
{
    let combinations: Vec<Vec<&K>> = sets.keys().into_iter().combinations(2).collect();
    if !do_parallel {
        combinations
            .iter()
            .map(|x| overlap_helper(x, sets))
            .collect()
    } else {
        combinations
            .into_par_iter()
            .map(|x| overlap_helper(&x, sets))
            .collect()
    }
}

// * Tests

#[ignore = "reason"]
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

#[test]
fn test_pairwise_overlap() {
    let mut map: HashMap<&str, HashSet<&str>> = HashMap::new();
    map.insert(
        "fruits",
        ["apple", "banana", "orange", "grape"]
            .iter()
            .map(|&s| s)
            .collect(),
    );
    map.insert(
        "vegetables",
        ["carrot", "broccoli", "spinach", "tomato"] // tomato overlaps
            .iter()
            .map(|&s| s)
            .collect(),
    );
    map.insert(
        "red_foods",
        ["apple", "tomato", "strawberry", "cherry"] // overlaps: apple, tomato
            .iter()
            .map(|&s| s)
            .collect(),
    );
    map.insert(
        "green_foods",
        ["spinach", "broccoli", "grape", "kiwi"] // overlaps: spinach, broccoli, grape
            .iter()
            .map(|&s| s)
            .collect(),
    );
    map.insert(
        "snacks",
        ["banana", "apple", "chips", "nuts"] // overlaps: banana, apple
            .iter()
            .map(|&s| s)
            .collect(),
    );
    map.insert(
        "juices",
        ["orange", "apple", "carrot", "grape"] // overlaps: orange, apple, carrot, grape
            .iter()
            .map(|&s| s)
            .collect(),
    );
    map.insert(
        "smoothies",
        ["banana", "strawberry", "spinach", "apple"] // overlaps: banana, strawberry, spinach, apple
            .iter()
            .map(|&s| s)
            .collect(),
    );
    let overlaps = pairwise_overlap_rs(&map, false);
    println!("{:?}", overlaps)
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
