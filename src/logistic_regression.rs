use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub fn sigmoid(z: Array2<f64>) -> Array2<f64> {
    1.0 / (1.0 + (-z).exp())
}

pub struct LogisticRegression {
    learning_rate: f64,
    n_iters: i32,
    eps: f64,
    w: Array2<f64>,
    b: f64
}

impl LogisticRegression {
    pub fn default() -> Self {
        LogisticRegression {
            learning_rate: 0.01,
            n_iters: 200,
            eps: 1e-12,
            w: Array2::zeros((1, 1)),
            b: 0.0
        }
    }

    pub fn new(learning_rate: f64, n_iters: i32) -> Self {
        LogisticRegression {
            learning_rate,
            n_iters,
            eps: 1e-12,
            w: Array2::zeros((1, 1)),
            b: 0.0
        }
    }

    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let z: Array2<f64> = x.dot(&self.w) + self.b;
        sigmoid(z)
    }

    #[allow(unused)]
    fn calculate_loss(&self, y: &Array2<f64>, a: &Array2<f64>) -> f64 {
        let ones = Array::ones(y.raw_dim());
        let loss: Array2<f64> = y * a.mapv(f64::ln) + (&ones - y) * (ones - a).mapv(f64::ln);
        -loss.mean().unwrap_or(0.0)
    }

    #[allow(unused_variables)]
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>) {
        // Xavier Initialization
        let bound = (6.0 / (x.shape()[1] as f64 + 1.0)).sqrt();
        self.w = Array2::random((x.shape()[1], 1), Uniform::new(-bound, bound));
        self.b = 0.0;

        for i in 1..self.n_iters + 1 {
            let a = self.forward(x);
            // Clipping
            let a = a.mapv(|x| x.max(self.eps).min(1.0 - self.eps));

            // let current_loss: f64 = self.calculate_loss(y, &a);
            // println!("Iter: {0}, Loss: {1}", i, current_loss);

            let a_y: Array2<f64> = a - y;
            let num_samples = x.shape()[0] as f64;
            let dw: Array2<f64> = x.t().dot(&a_y) / num_samples + (0.01 / num_samples) * &self.w;
            let db: f64 = a_y.mean().unwrap();

            self.w = &self.w - self.learning_rate * dw;
            self.b = self.b - self.learning_rate * db;
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array2<f64> {
        let a: Array2<f64> = self.forward(x);
        a.mapv(|i| if i < 0.5 { 0.0 } else { 1.0 })
    }

    pub fn score(&self, x: &Array2<f64>, y: &Array2<f64>) -> f64 {
        let class_pred = self.predict(x);
        let correct_predictions = class_pred
            .iter()
            .zip(y.iter())
            .filter(|(pred, actual)| pred == actual)
            .count();

        let total_predictions = y.len();
        let accuracy = correct_predictions as f64 / total_predictions as f64;

        accuracy
    }
}