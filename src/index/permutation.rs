use crate::dim::{Const, Dim};
use crate::layout::{Layout, Strided};

/// Array permutation trait, for array types after permutation of dimensions.
pub trait Permutation {
    /// Layout after permuting dimensions.
    type Layout<L: Layout>: Layout;
}

impl Permutation for (Const<0>,) {
    type Layout<L: Layout> = L;
}

macro_rules! impl_permutation {
    ($n:tt, ($($pre:tt),*), ($($post:tt),*), $layout:ty) => {
        impl<$($pre: Dim,)* $($post: Dim),*> Permutation for ($($pre,)* Const<$n>, $($post),*)
        where
            ($($pre,)* $($post,)*): Permutation
        {
            type Layout<L: Layout> = $layout;
        }
    };
}

impl_permutation!(1, (), (X), Strided);
impl_permutation!(1, (X), (), L);

impl_permutation!(2, (), (X, Y), Strided);
impl_permutation!(2, (X), (Y), <(X, Y) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(2, (X, Y), (), <(X, Y) as Permutation>::Layout<L>);

impl_permutation!(3, (), (X, Y, Z), Strided);
impl_permutation!(3, (X), (Y, Z), <(X, Y, Z) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(3, (X, Y), (Z), <(X, Y, Z) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(3, (X, Y, Z), (), <(X, Y, Z) as Permutation>::Layout<L>);

impl_permutation!(4, (), (X, Y, Z, W), Strided);
impl_permutation!(4, (X), (Y, Z, W), <(X, Y, Z, W) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(4, (X, Y), (Z, W), <(X, Y, Z, W) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(4, (X, Y, Z), (W), <(X, Y, Z, W) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(4, (X, Y, Z, W), (), <(X, Y, Z, W) as Permutation>::Layout<L>);

impl_permutation!(5, (), (X, Y, Z, W, U), Strided);
impl_permutation!(5, (X), (Y, Z, W, U), <(X, Y, Z, W, U) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(5, (X, Y), (Z, W, U), <(X, Y, Z, W, U) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(5, (X, Y, Z), (W, U), <(X, Y, Z, W, U) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(5, (X, Y, Z, W), (U), <(X, Y, Z, W, U) as Permutation>::Layout<L::NonUniform>);
impl_permutation!(5, (X, Y, Z, W, U), (), <(X, Y, Z, W, U) as Permutation>::Layout<L>);
