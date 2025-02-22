use ndarray::{arr2, s, Array1, Array2, ArrayBase, ArrayView1};
use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use rayon::prelude::*;

fn phi_proportionality(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    let log = x.ln();
    (&log - y.ln()).var(1.0) / log.var(1.0)
}

/// Perform a calculation pairwise across all features in the matrix
///
/// # Arguments
/// * arr : a sample x feature matrix
///
fn do_pairwise<F>(arr: &Array2<f64>, do_parallel: bool, calc_fn: F) -> Array2<f64>
where
    F: Fn(ArrayView1<f64>, ArrayView1<f64>) -> f64 + Sync + 'static,
{
    let ncols = arr.ncols();
    let mut result = Array2::zeros([ncols, ncols]);
    if !do_parallel {
        for i in 0..ncols {
            let calculated_row: Array1<f64> = (0..ncols)
                .map(|j| calc_fn(arr.slice(s![.., i]), arr.slice(s![.., j])))
                .collect();
            result.slice_mut(s![.., i]).assign(&calculated_row);
        }
    } else {
        for i in 0..ncols {
            let calculated_row: Vec<f64> = (0..ncols)
                .into_par_iter()
                .map(|j| calc_fn(arr.slice(s![.., i]), arr.slice(s![.., j])))
                .collect();
            let row: Array1<f64> = ArrayBase::from_vec(calculated_row);
            result.slice_mut(s![.., i]).assign(&row);
        }
    }
    result
}

#[test]
fn test_phi_prop() {
    let mut h = arr2(&[[8, 1, 2, 3], [4, 5, 6, 7]]).mapv(|x| x as f64);
    let dim = h.raw_dim();
    let s1 = h.slice(s![0, ..]);
    let s2 = h.slice(s![1, ..]);
    let phi = do_pairwise(&h, true, phi_proportionality);
    println!("{:?}", phi);
}
