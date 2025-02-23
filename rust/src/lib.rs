use core::f64;

use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, ArrayViewD, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::types::PyModuleMethods;
use pyo3::wrap_pyfunction;
use pyo3::{pyfunction, pymodule, types::PyModule, Bound, PyResult, Python};
use rayon::{prelude::*, result};

#[pymodule]
fn _rust_helpers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(phi_matrix, m)?)?;
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

fn phi_proportionality_test(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    let log = x.ln();
    (&log - y.ln()).var(1.) / log.var(1.)
}

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

/// Perform a calculation pairwise across all features in the matrix
///
/// # Arguments
/// * arr : a sample x feature matrix
///
fn do_pairwise<F>(arr: ArrayView2<f64>, do_parallel: bool, calc_fn: F) -> Array2<f64>
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
    use ndarray::arr2;
    let h = arr2(&[[8, 1, 2, 3], [4, 5, 6, 7]]).mapv(|x| x as f64);
    let dim = h.raw_dim();
    let s1 = h.slice(s![0, ..]);
    let s2 = h.slice(s![1, ..]);
    let phi = do_pairwise(h.view(), true, phi_proportionality_test);
    let phi2 = phi_proportionality_rs(h.view(), false);
    println!("{:?}", phi);
    println!("{:?}", phi2);
}
