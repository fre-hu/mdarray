#!/bin/sh

set -ex

cargo clippy --features serde
cargo clippy --features nightly --features serde

cargo doc --features serde
cargo doc --features nightly --features serde

cargo fmt

cargo test
cargo test --features serde
cargo test --features nightly
cargo test --features nightly --features serde

cargo test --no-default-features
cargo test --features serde --no-default-features
cargo test --features nightly --no-default-features
cargo test --features nightly --features serde --no-default-features

MIRIFLAGS="-Zmiri-tree-borrows" cargo miri test --features serde
MIRIFLAGS="-Zmiri-tree-borrows" cargo miri test --features nightly --features serde
