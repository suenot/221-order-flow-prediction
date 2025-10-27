use order_flow_prediction::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Order Flow Prediction - Trading Example ===\n");

    // ── Step 1: Fetch live data from Bybit ──────────────────────────
    println!("[1] Fetching BTCUSDT data from Bybit V5 API...\n");

    let client = BybitClient::new();

    let klines = match client.get_klines("BTCUSDT", "1", 50).await {
        Ok(k) => {
            println!("  Fetched {} kline bars", k.len());
            if let Some(last) = k.last() {
                println!(
                    "  Latest bar: O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
                    last.open, last.high, last.low, last.close, last.volume
                );
            }
            k
        }
        Err(e) => {
            println!("  Could not fetch klines: {}. Using synthetic data.", e);
            Vec::new()
        }
    };

    let orderbook = match client.get_orderbook("BTCUSDT", 25).await {
        Ok((bids, asks)) => {
            println!(
                "  Order book: {} bid levels, {} ask levels",
                bids.len(),
                asks.len()
            );
            if let (Some(best_bid), Some(best_ask)) = (bids.first(), asks.first()) {
                println!(
                    "  Best bid: {:.2} ({:.4}), Best ask: {:.2} ({:.4})",
                    best_bid.0, best_bid.1, best_ask.0, best_ask.1
                );
            }
            Some((bids, asks))
        }
        Err(e) => {
            println!("  Could not fetch orderbook: {}. Using synthetic data.", e);
            None
        }
    };

    // ── Step 2: Compute OFI ─────────────────────────────────────────
    println!("\n[2] Computing Order Flow Imbalance (OFI)...\n");

    let mut ofi_tracker = OrderFlowImbalance::new();

    // Use real order book or synthetic snapshots
    let snapshots = if let Some((ref bids, ref asks)) = orderbook {
        // Create a snapshot from real data
        if let (Some(bb), Some(ba)) = (bids.first(), asks.first()) {
            vec![
                BookSnapshot {
                    bid_price: bb.0,
                    bid_volume: bb.1,
                    ask_price: ba.0,
                    ask_volume: ba.1,
                },
                // Simulate a second snapshot with slight changes
                BookSnapshot {
                    bid_price: bb.0,
                    bid_volume: bb.1 * 1.05,
                    ask_price: ba.0,
                    ask_volume: ba.1 * 0.95,
                },
            ]
        } else {
            generate_synthetic_snapshots(20, 50000.0)
        }
    } else {
        generate_synthetic_snapshots(20, 50000.0)
    };

    for snap in &snapshots {
        if let Some(ofi_val) = ofi_tracker.update(*snap) {
            println!(
                "  OFI = {:+.4} | Bid: {:.2} ({:.4}) | Ask: {:.2} ({:.4})",
                ofi_val, snap.bid_price, snap.bid_volume, snap.ask_price, snap.ask_volume
            );
        }
    }

    let cum_ofi = ofi_tracker.cumulative(10);
    println!("  Cumulative OFI (last 10): {:+.4}", cum_ofi);

    // ── Step 3: Compute Cumulative Delta ────────────────────────────
    println!("\n[3] Computing Cumulative Delta...\n");

    let mut cum_delta = CumulativeDelta::new();

    // Use kline data to approximate trades or generate synthetic
    if !klines.is_empty() {
        for (i, kline) in klines.iter().enumerate() {
            let is_buy = kline.close >= kline.open;
            cum_delta.add_trade(kline.volume, is_buy);
            if i >= klines.len() - 5 {
                println!(
                    "  Bar {}: vol={:.2}, side={}, delta={:+.2}",
                    i,
                    kline.volume,
                    if is_buy { "BUY" } else { "SELL" },
                    cum_delta.value()
                );
            }
        }
    } else {
        let trades = generate_synthetic_trades(50, 50000.0);
        for (_price, vol, is_buy) in &trades {
            cum_delta.add_trade(*vol, *is_buy);
        }
        let _ = trades; // suppress unused warning
    }

    println!("  Final cumulative delta: {:+.4}", cum_delta.value());
    println!("  Trend (last 10): {:+.4}", cum_delta.trend(10));

    // ── Step 4: Compute VPIN ────────────────────────────────────────
    println!("\n[4] Computing VPIN...\n");

    let mut vpin_calc = VPINCalculator::new(10.0, 5);

    let trades = generate_synthetic_trades(200, 50000.0);
    for (price, vol, _) in &trades {
        vpin_calc.add_trade(*price, *vol);
    }

    println!("  Completed buckets: {}", vpin_calc.completed_buckets());
    match vpin_calc.vpin() {
        Some(v) => println!("  VPIN = {:.4} (0=balanced, 1=fully informed)", v),
        None => println!("  VPIN: not enough buckets yet"),
    }

    // ── Step 5: Train classifier and predict ────────────────────────
    println!("\n[5] Training Order Flow Classifier...\n");

    let training_data = generate_training_data(2000);
    let (train, test) = training_data.split_at(1600);

    let mut classifier = OrderFlowClassifier::new(5, 0.01);
    println!("  Accuracy before training: {:.2}%", classifier.accuracy(test) * 100.0);

    classifier.train(&train.to_vec(), 100);
    let acc = classifier.accuracy(test);
    println!("  Accuracy after training:  {:.2}%", acc * 100.0);
    println!("  Weights: {:?}", classifier.weights());
    println!("  Bias: {:.4}", classifier.bias());

    // ── Step 6: Make a live prediction ──────────────────────────────
    println!("\n[6] Live Prediction...\n");

    let vpin_value = vpin_calc.vpin().unwrap_or(0.5);
    let spread = if let Some((ref bids, ref asks)) = orderbook {
        if let (Some(bb), Some(ba)) = (bids.first(), asks.first()) {
            ba.0 - bb.0
        } else {
            1.0
        }
    } else {
        1.0
    };

    let imbalance = if let Some(snap) = snapshots.last() {
        bid_ask_imbalance(snap.bid_volume, snap.ask_volume)
    } else {
        0.0
    };

    let features = vec![cum_ofi, cum_delta.value(), vpin_value, spread, imbalance];
    println!("  Features: OFI={:.4}, Delta={:.4}, VPIN={:.4}, Spread={:.4}, Imbalance={:.4}",
        features[0], features[1], features[2], features[3], features[4]);

    let (direction, confidence) = classifier.predict(&features);
    println!(
        "  Prediction: {} with {:.1}% confidence",
        if direction { "UP" } else { "DOWN" },
        confidence * 100.0
    );

    println!("\n=== Done ===");
    Ok(())
}
