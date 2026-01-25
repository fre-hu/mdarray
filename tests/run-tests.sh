#!/bin/sh

export MIRIFLAGS="-Zmiri-tree-borrows"
export RUSTFLAGS="-D warnings"

export RUSTUP_TOOLCHAIN="1.89"

set -ex

cargo clippy --features serde
cargo doc --features serde

cargo test
cargo test --features serde
cargo test --no-default-features
cargo test --features serde --no-default-features

cargo +nightly clippy --features nightly --features serde
cargo +nightly doc --features nightly --features serde

cargo +nightly fmt

cargo +nightly test --features nightly
cargo +nightly test --features nightly --features serde
cargo +nightly test --features nightly --no-default-features
cargo +nightly test --features nightly --features serde --no-default-features

cargo +nightly miri test --features nightly --features serde
