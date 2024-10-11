mod logistic_regression;

use logistic_regression::LogisticRegression;
use ndarray::Array2;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};


fn standardize(x: Array2<f64>) -> Array2<f64> {
    let mut x_standardized = x.clone();
    for mut column in x_standardized.columns_mut() {
        let mean = column.mean().unwrap();
        let std_dev = column.std(0.0);
        column.mapv_inplace(|x| (x - mean) / std_dev);
    }
    x_standardized
}

#[allow(unused_variables)]
fn read_dataset(file_path: &str) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut target: Vec<f64> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let mut row: Vec<f64> = Vec::new();

        for (i, item) in line.split(',').enumerate() {
            if i == 1 {
                match item {
                    "M" => target.push(1.0),
                    "B" => target.push(0.0),
                    _ => return Err(From::from("Unexpected value in the target column")),
                }
            } else {
                row.push(item.parse().unwrap_or(0.0));
            }
        }
        features.push(row);
    }

    let n_rows = features.len();
    let n_cols = features[0].len();

    let flat_features: Vec<f64> = features.into_iter().flatten().collect();
    let mut x: Array2<f64> = Array2::from_shape_vec((n_rows, n_cols), flat_features)?;
    x = standardize(x);
    let y: Array2<f64> = Array2::from_shape_vec((n_rows, 1), target)?;

    Ok((x, y))
}

#[allow(unused)]
fn create_dummy_dataset() -> (Array2<f64>, Array2<f64>) {
    let x: Array2<f64> = Array2::from_shape_vec(
        (4, 3),
        vec![0.3, 0.2, 1.0, 0.9, 1.0, 0.9, 0.7, 0.1, 1.0, 0.5, 0.3, 0.2],
    ).unwrap();

    let y: Array2<f64> = Array2::from_shape_vec(
        (4, 1),
        vec![0.0, 1.0, 0.0, 1.0],
    ).unwrap();

    (x, y)
}

#[allow(unused)]
fn test_dataset(data_path: &str) {
    match read_dataset(data_path) {
        Ok((x, y)) => {
            let mut model = LogisticRegression::new(0.02, 200);
            model.fit(&x, &y);
            let accuracy = model.score(&x, &y);
            println!("Accuracy: {}", accuracy);
        }
        Err(e) => {
            eprintln!("An error occurred: {}", e);
        }
    }
}

#[allow(unused)]
fn test_dummy_dataset() {
    let (x, y) = create_dummy_dataset();
    let mut model = LogisticRegression::default();
    model.fit(&x, &y);
    let accuracy = model.score(&x, &y);
    println!("Accuracy: {}", accuracy);
}

fn main() {
    let data_path = "resources/wdbc.data";
    test_dataset(data_path)
}
