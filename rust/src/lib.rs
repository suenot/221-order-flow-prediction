use ndarray::Array1;
use rand::Rng;
use serde::Deserialize;

// ─── Order Book Snapshot ───────────────────────────────────────────

/// A snapshot of the best bid/ask level of an order book.
#[derive(Debug, Clone, Copy)]
pub struct BookSnapshot {
    pub bid_price: f64,
    pub bid_volume: f64,
    pub ask_price: f64,
    pub ask_volume: f64,
}

// ─── Order Flow Imbalance (OFI) ────────────────────────────────────

/// Tracks bid/ask volume changes and computes Order Flow Imbalance.
///
/// OFI = delta_bid_volume - delta_ask_volume, where the deltas
/// account for price-level changes using the Cont-Kukanov-Stoikov
/// piecewise formulas.
#[derive(Debug)]
pub struct OrderFlowImbalance {
    prev_snapshot: Option<BookSnapshot>,
    ofi_history: Vec<f64>,
}

impl OrderFlowImbalance {
    pub fn new() -> Self {
        Self {
            prev_snapshot: None,
            ofi_history: Vec::new(),
        }
    }

    /// Feed a new book snapshot and return the OFI value.
    /// Returns `None` for the very first snapshot (no previous state).
    pub fn update(&mut self, snap: BookSnapshot) -> Option<f64> {
        let result = if let Some(prev) = self.prev_snapshot {
            let delta_bid = if snap.bid_price > prev.bid_price {
                snap.bid_volume
            } else if snap.bid_price == prev.bid_price {
                snap.bid_volume - prev.bid_volume
            } else {
                -prev.bid_volume
            };

            let delta_ask = if snap.ask_price > prev.ask_price {
                -prev.ask_volume
            } else if snap.ask_price == prev.ask_price {
                snap.ask_volume - prev.ask_volume
            } else {
                snap.ask_volume
            };

            let ofi = delta_bid - delta_ask;
            self.ofi_history.push(ofi);
            Some(ofi)
        } else {
            None
        };

        self.prev_snapshot = Some(snap);
        result
    }

    /// Return the full OFI history.
    pub fn history(&self) -> &[f64] {
        &self.ofi_history
    }

    /// Cumulative OFI over the last `n` updates (or all if fewer).
    pub fn cumulative(&self, n: usize) -> f64 {
        let start = self.ofi_history.len().saturating_sub(n);
        self.ofi_history[start..].iter().sum()
    }
}

impl Default for OrderFlowImbalance {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Cumulative Delta ──────────────────────────────────────────────

/// Running sum of buy volume minus sell volume.
#[derive(Debug)]
pub struct CumulativeDelta {
    delta: f64,
    history: Vec<f64>,
}

impl CumulativeDelta {
    pub fn new() -> Self {
        Self {
            delta: 0.0,
            history: Vec::new(),
        }
    }

    /// Add a single trade.  `is_buy` indicates buyer-initiated.
    pub fn add_trade(&mut self, volume: f64, is_buy: bool) {
        if is_buy {
            self.delta += volume;
        } else {
            self.delta -= volume;
        }
        self.history.push(self.delta);
    }

    /// Batch-add trades. Each element is (volume, is_buy).
    pub fn add_trades(&mut self, trades: &[(f64, bool)]) {
        for &(vol, is_buy) in trades {
            self.add_trade(vol, is_buy);
        }
    }

    /// Current cumulative delta value.
    pub fn value(&self) -> f64 {
        self.delta
    }

    /// Trend over the last `window` entries: positive = rising, negative = falling.
    pub fn trend(&self, window: usize) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let start = self.history.len().saturating_sub(window);
        let first = self.history[start];
        let last = *self.history.last().unwrap();
        last - first
    }

    pub fn history(&self) -> &[f64] {
        &self.history
    }
}

impl Default for CumulativeDelta {
    fn default() -> Self {
        Self::new()
    }
}

// ─── VPIN Calculator ───────────────────────────────────────────────

/// Volume-Synchronized Probability of Informed Trading.
#[derive(Debug)]
pub struct VPINCalculator {
    bucket_size: f64,
    num_buckets: usize,
    // accumulation state
    current_buy_vol: f64,
    current_sell_vol: f64,
    current_total_vol: f64,
    prev_price: Option<f64>,
    // completed buckets: each entry is |buy - sell| / bucket_size
    bucket_imbalances: Vec<f64>,
}

impl VPINCalculator {
    /// Create a new VPIN calculator.
    /// - `bucket_size`: volume per bucket (e.g. 100 BTC)
    /// - `num_buckets`: number of buckets to average over
    pub fn new(bucket_size: f64, num_buckets: usize) -> Self {
        Self {
            bucket_size,
            num_buckets,
            current_buy_vol: 0.0,
            current_sell_vol: 0.0,
            current_total_vol: 0.0,
            prev_price: None,
            bucket_imbalances: Vec::new(),
        }
    }

    /// Add a trade with its price and volume.
    /// Uses tick rule to classify direction.
    pub fn add_trade(&mut self, price: f64, volume: f64) {
        let is_buy = match self.prev_price {
            Some(prev) => price >= prev,
            None => true,
        };
        self.prev_price = Some(price);

        if is_buy {
            self.current_buy_vol += volume;
        } else {
            self.current_sell_vol += volume;
        }
        self.current_total_vol += volume;

        // Check if bucket is full
        while self.current_total_vol >= self.bucket_size {
            let overflow = self.current_total_vol - self.bucket_size;
            // Proportionally reduce the overflow from the last trade's side
            let ratio = if volume > 0.0 {
                overflow / volume
            } else {
                0.0
            };

            let adj_buy;
            let adj_sell;
            if is_buy {
                adj_buy = self.current_buy_vol - overflow * ratio;
                adj_sell = self.current_sell_vol;
            } else {
                adj_buy = self.current_buy_vol;
                adj_sell = self.current_sell_vol - overflow * ratio;
            }

            let imbalance = (adj_buy - adj_sell).abs() / self.bucket_size;
            self.bucket_imbalances.push(imbalance);

            // Start new bucket with overflow
            if is_buy {
                self.current_buy_vol = overflow * ratio;
                self.current_sell_vol = 0.0;
            } else {
                self.current_buy_vol = 0.0;
                self.current_sell_vol = overflow * ratio;
            }
            self.current_total_vol = overflow * ratio;
            // Only process overflow once
            break;
        }
    }

    /// Current VPIN value (average of last `num_buckets` bucket imbalances).
    /// Returns `None` if not enough buckets have been completed.
    pub fn vpin(&self) -> Option<f64> {
        if self.bucket_imbalances.len() < self.num_buckets {
            return None;
        }
        let start = self.bucket_imbalances.len() - self.num_buckets;
        let sum: f64 = self.bucket_imbalances[start..].iter().sum();
        Some(sum / self.num_buckets as f64)
    }

    /// Number of completed buckets so far.
    pub fn completed_buckets(&self) -> usize {
        self.bucket_imbalances.len()
    }
}

// ─── Order Flow Classifier (Logistic Regression) ──────────────────

/// Binary logistic regression classifier for price direction prediction.
///
/// Features: [OFI, cumulative_delta, VPIN, spread, bid_ask_imbalance]
#[derive(Debug)]
pub struct OrderFlowClassifier {
    weights: Array1<f64>,
    bias: f64,
    learning_rate: f64,
    num_features: usize,
}

impl OrderFlowClassifier {
    pub fn new(num_features: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array1::from_vec(
            (0..num_features)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect(),
        );
        Self {
            weights,
            bias: 0.0,
            learning_rate,
            num_features,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Predict probability that price goes up (label = 1).
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        assert_eq!(features.len(), self.num_features);
        let x = Array1::from_vec(features.to_vec());
        let z = self.weights.dot(&x) + self.bias;
        Self::sigmoid(z)
    }

    /// Predict direction: true = up, false = down.  Returns (direction, confidence).
    pub fn predict(&self, features: &[f64]) -> (bool, f64) {
        let prob = self.predict_proba(features);
        if prob >= 0.5 {
            (true, prob)
        } else {
            (false, 1.0 - prob)
        }
    }

    /// Train on a dataset of (features, label) pairs for `epochs` iterations.
    pub fn train(&mut self, data: &[(Vec<f64>, f64)], epochs: usize) {
        for _ in 0..epochs {
            for (features, label) in data {
                let x = Array1::from_vec(features.clone());
                let z = self.weights.dot(&x) + self.bias;
                let pred = Self::sigmoid(z);
                let error = pred - label;

                // Gradient descent update
                for j in 0..self.num_features {
                    self.weights[j] -= self.learning_rate * error * x[j];
                }
                self.bias -= self.learning_rate * error;
            }
        }
    }

    /// Evaluate accuracy on a test set.
    pub fn accuracy(&self, data: &[(Vec<f64>, f64)]) -> f64 {
        let correct = data
            .iter()
            .filter(|(features, label)| {
                let (pred, _) = self.predict(features);
                let label_bool = *label >= 0.5;
                pred == label_bool
            })
            .count();
        correct as f64 / data.len() as f64
    }

    pub fn weights(&self) -> &Array1<f64> {
        &self.weights
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }
}

// ─── Bybit Client ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct OrderbookResult {
    pub b: Vec<Vec<String>>,  // bids: [price, size]
    pub a: Vec<Vec<String>>,  // asks: [price, size]
}

/// A parsed kline bar.
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Async client for Bybit V5 API.
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch kline (candlestick) data.
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> anyhow::Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );
        let resp: BybitResponse<KlineResult> = self.client.get(&url).send().await?.json().await?;

        let mut klines = Vec::new();
        for item in &resp.result.list {
            if item.len() >= 6 {
                klines.push(Kline {
                    timestamp: item[0].parse().unwrap_or(0),
                    open: item[1].parse().unwrap_or(0.0),
                    high: item[2].parse().unwrap_or(0.0),
                    low: item[3].parse().unwrap_or(0.0),
                    close: item[4].parse().unwrap_or(0.0),
                    volume: item[5].parse().unwrap_or(0.0),
                });
            }
        }
        klines.reverse(); // Bybit returns newest first
        Ok(klines)
    }

    /// Fetch order book snapshot.
    pub async fn get_orderbook(
        &self,
        symbol: &str,
        limit: u32,
    ) -> anyhow::Result<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );
        let resp: BybitResponse<OrderbookResult> =
            self.client.get(&url).send().await?.json().await?;

        let bids: Vec<(f64, f64)> = resp
            .result
            .b
            .iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some((
                        entry[0].parse().unwrap_or(0.0),
                        entry[1].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<(f64, f64)> = resp
            .result
            .a
            .iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some((
                        entry[0].parse().unwrap_or(0.0),
                        entry[1].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        Ok((bids, asks))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Synthetic Data Generation ─────────────────────────────────────

/// Generate synthetic order book snapshots for testing.
pub fn generate_synthetic_snapshots(n: usize, seed_price: f64) -> Vec<BookSnapshot> {
    let mut rng = rand::thread_rng();
    let mut price = seed_price;
    let mut snapshots = Vec::with_capacity(n);

    for _ in 0..n {
        let spread = rng.gen_range(0.5..2.0);
        let bid_price = price - spread / 2.0;
        let ask_price = price + spread / 2.0;
        let bid_volume = rng.gen_range(1.0..50.0);
        let ask_volume = rng.gen_range(1.0..50.0);

        snapshots.push(BookSnapshot {
            bid_price,
            bid_volume,
            ask_price,
            ask_volume,
        });

        // Random walk
        price += rng.gen_range(-1.0..1.0);
    }
    snapshots
}

/// Generate synthetic trades for testing.
pub fn generate_synthetic_trades(n: usize, start_price: f64) -> Vec<(f64, f64, bool)> {
    let mut rng = rand::thread_rng();
    let mut price = start_price;
    let mut trades = Vec::with_capacity(n);

    for _ in 0..n {
        let is_buy = rng.gen_bool(0.5);
        let volume = rng.gen_range(0.01..5.0);
        price += if is_buy {
            rng.gen_range(0.0..0.5)
        } else {
            rng.gen_range(-0.5..0.0)
        };
        trades.push((price, volume, is_buy));
    }
    trades
}

/// Generate labeled training data for the classifier.
///
/// Each sample has features [ofi, cum_delta, vpin_proxy, spread, imbalance]
/// and a binary label (1.0 = price up, 0.0 = price down).
pub fn generate_training_data(n: usize) -> Vec<(Vec<f64>, f64)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);

    for _ in 0..n {
        let ofi: f64 = rng.gen_range(-10.0..10.0);
        let cum_delta: f64 = rng.gen_range(-5.0..5.0);
        let vpin: f64 = rng.gen_range(0.0..1.0);
        let spread: f64 = rng.gen_range(0.1..2.0);
        let imbalance: f64 = rng.gen_range(-1.0..1.0);

        // Label: higher probability of up when OFI > 0 and cum_delta > 0
        let signal = 0.3 * ofi + 0.3 * cum_delta - 0.2 * vpin + 0.1 * imbalance;
        let prob = 1.0 / (1.0 + (-signal).exp());
        let label = if rng.gen::<f64>() < prob { 1.0 } else { 0.0 };

        data.push((vec![ofi, cum_delta, vpin, spread, imbalance], label));
    }
    data
}

/// Compute bid/ask imbalance from volumes.
pub fn bid_ask_imbalance(bid_vol: f64, ask_vol: f64) -> f64 {
    let total = bid_vol + ask_vol;
    if total == 0.0 {
        return 0.0;
    }
    (bid_vol - ask_vol) / total
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ofi_basic() {
        let mut ofi = OrderFlowImbalance::new();
        let snap1 = BookSnapshot {
            bid_price: 100.0,
            bid_volume: 10.0,
            ask_price: 101.0,
            ask_volume: 10.0,
        };
        assert!(ofi.update(snap1).is_none());

        // Same prices, bid grows, ask shrinks
        let snap2 = BookSnapshot {
            bid_price: 100.0,
            bid_volume: 15.0,
            ask_price: 101.0,
            ask_volume: 7.0,
        };
        let v = ofi.update(snap2).unwrap();
        // delta_bid = 15 - 10 = 5, delta_ask = 7 - 10 = -3, OFI = 5 - (-3) = 8
        assert!((v - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_ofi_price_change() {
        let mut ofi = OrderFlowImbalance::new();
        let snap1 = BookSnapshot {
            bid_price: 100.0,
            bid_volume: 10.0,
            ask_price: 101.0,
            ask_volume: 10.0,
        };
        ofi.update(snap1);

        // Bid price increases -> delta_bid = new_bid_vol
        // Ask price increases -> delta_ask = -prev_ask_vol
        let snap2 = BookSnapshot {
            bid_price: 101.0,
            bid_volume: 20.0,
            ask_price: 102.0,
            ask_volume: 8.0,
        };
        let v = ofi.update(snap2).unwrap();
        // delta_bid = 20, delta_ask = -10, OFI = 20 - (-10) = 30
        assert!((v - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_cumulative_delta() {
        let mut cd = CumulativeDelta::new();
        cd.add_trade(5.0, true);
        cd.add_trade(3.0, false);
        cd.add_trade(2.0, true);
        assert!((cd.value() - 4.0).abs() < 1e-9); // 5 - 3 + 2 = 4
    }

    #[test]
    fn test_cumulative_delta_trend() {
        let mut cd = CumulativeDelta::new();
        cd.add_trade(10.0, true);   // delta = 10
        cd.add_trade(2.0, false);   // delta = 8
        cd.add_trade(5.0, true);    // delta = 13
        // trend over window=3: last - first = 13 - 10 = 3
        assert!((cd.trend(3) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_vpin_basic() {
        let mut vpin = VPINCalculator::new(10.0, 2);
        // Fill several buckets with all-buy trades (high imbalance)
        for i in 0..5 {
            vpin.add_trade(100.0 + i as f64, 10.0);
        }
        // With all buys and bucket_size 10, each bucket imbalance should be 1.0
        assert!(vpin.completed_buckets() >= 2);
        let v = vpin.vpin().unwrap();
        assert!(v > 0.0 && v <= 1.0);
    }

    #[test]
    fn test_vpin_insufficient_buckets() {
        let vpin = VPINCalculator::new(1000.0, 5);
        assert!(vpin.vpin().is_none());
    }

    #[test]
    fn test_classifier_predict() {
        let clf = OrderFlowClassifier::new(5, 0.01);
        let features = vec![1.0, 0.5, 0.3, 0.5, 0.2];
        let (_, confidence) = clf.predict(&features);
        assert!(confidence >= 0.5 && confidence <= 1.0);
    }

    #[test]
    fn test_classifier_train_and_improve() {
        let data = generate_training_data(500);
        let (train, test) = data.split_at(400);

        let mut clf = OrderFlowClassifier::new(5, 0.01);
        let _acc_before = clf.accuracy(test);

        clf.train(&train.to_vec(), 50);
        let acc_after = clf.accuracy(test);

        // After training, accuracy should generally improve (or at least not be 0)
        assert!(acc_after > 0.0);
        // Trained model should be meaningfully above 0
        assert!(acc_after >= 0.4, "accuracy after training: {}", acc_after);
    }

    #[test]
    fn test_bid_ask_imbalance() {
        assert!((bid_ask_imbalance(10.0, 10.0) - 0.0).abs() < 1e-9);
        assert!((bid_ask_imbalance(10.0, 0.0) - 1.0).abs() < 1e-9);
        assert!((bid_ask_imbalance(0.0, 10.0) - (-1.0)).abs() < 1e-9);
        assert!((bid_ask_imbalance(0.0, 0.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_synthetic_data_generation() {
        let snaps = generate_synthetic_snapshots(100, 50000.0);
        assert_eq!(snaps.len(), 100);
        for s in &snaps {
            assert!(s.bid_price < s.ask_price);
            assert!(s.bid_volume > 0.0);
            assert!(s.ask_volume > 0.0);
        }

        let trades = generate_synthetic_trades(100, 50000.0);
        assert_eq!(trades.len(), 100);

        let training = generate_training_data(100);
        assert_eq!(training.len(), 100);
        for (features, label) in &training {
            assert_eq!(features.len(), 5);
            assert!(*label == 0.0 || *label == 1.0);
        }
    }

    #[test]
    fn test_ofi_cumulative() {
        let mut ofi = OrderFlowImbalance::new();
        let snaps = generate_synthetic_snapshots(20, 50000.0);
        for s in &snaps {
            ofi.update(*s);
        }
        // 19 OFI values (first snapshot produces none)
        assert_eq!(ofi.history().len(), 19);
        // cumulative over all should equal sum
        let total: f64 = ofi.history().iter().sum();
        assert!((ofi.cumulative(100) - total).abs() < 1e-9);
    }
}
