use std::fmt::Debug;

use std::ops::{
    Bound, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use crate::layout::{Dense, Layout};

/// Array dimension trait, for rank and types for shape and strides.
pub trait Dim {
    /// Next higher dimension.
    type Higher: Dim;

    /// Next lower dimension.
    type Lower: Dim;

    /// Corresponding layout based on the dimension.
    type Layout<L: Layout>: Layout;

    /// Array shape type.
    type Shape: Shape<Dim = Self>;

    /// Array strides type.
    type Strides: Strides<Dim = Self>;

    /// Array rank, i.e. the number of dimensions.
    const RANK: usize;

    #[doc(hidden)]
    fn add_dim(shape: Self::Shape, size: usize) -> <Self::Higher as Dim>::Shape {
        assert!(<Self::Higher as Dim>::RANK > Self::RANK, "invalid rank");

        let mut new_shape = <Self::Higher as Dim>::Shape::default();

        new_shape[..Self::RANK].copy_from_slice(&shape[..]);
        new_shape[Self::RANK] = size;
        new_shape
    }

    #[doc(hidden)]
    fn checked_len(shape: Self::Shape) -> usize {
        shape[..].iter().fold(1, |acc, &x| acc.checked_mul(x).expect("length too large"))
    }

    #[doc(hidden)]
    fn remove_dim(shape: Self::Shape, dim: usize) -> <Self::Lower as Dim>::Shape {
        assert!(dim < Self::RANK, "invalid dimension");

        let mut new_shape = <Self::Lower as Dim>::Shape::default();

        new_shape[..dim].copy_from_slice(&shape[..dim]);
        new_shape[dim..].copy_from_slice(&shape[dim + 1..]);
        new_shape
    }

    #[doc(hidden)]
    fn resize_dim(mut shape: Self::Shape, dim: usize, new_size: usize) -> Self::Shape {
        assert!(dim < Self::RANK, "invalid dimension");

        shape[dim] = new_size;
        shape
    }
}

/// Array shape trait.
pub trait Shape:
    Copy
    + Debug
    + Default
    + IndexMut<(Bound<usize>, Bound<usize>), Output = [usize]>
    + IndexMut<usize, Output = usize>
    + IndexMut<Range<usize>, Output = [usize]>
    + IndexMut<RangeFrom<usize>, Output = [usize]>
    + IndexMut<RangeFull, Output = [usize]>
    + IndexMut<RangeInclusive<usize>, Output = [usize]>
    + IndexMut<RangeTo<usize>, Output = [usize]>
    + IndexMut<RangeToInclusive<usize>, Output = [usize]>
{
    /// Array dimension type.
    type Dim: Dim<Shape = Self>;
}

/// Array strides trait.
pub trait Strides:
    Copy
    + Debug
    + Default
    + IndexMut<(Bound<usize>, Bound<usize>), Output = [isize]>
    + IndexMut<usize, Output = isize>
    + IndexMut<Range<usize>, Output = [isize]>
    + IndexMut<RangeFrom<usize>, Output = [isize]>
    + IndexMut<RangeFull, Output = [isize]>
    + IndexMut<RangeInclusive<usize>, Output = [isize]>
    + IndexMut<RangeTo<usize>, Output = [isize]>
    + IndexMut<RangeToInclusive<usize>, Output = [isize]>
{
    /// Array dimension type.
    type Dim: Dim<Strides = Self>;
}

/// Type-level constant.
pub struct Const<const N: usize>;

macro_rules! impl_dim {
    (($($n:tt),*), ($($layout:ty),*)) => {
        $(
            impl Dim for Const<$n> {
                type Higher = Const<{ $n + ($n < 6) as usize }>;
                type Lower = Const<{ $n - ($n > 0) as usize }>;

                type Layout<L: Layout> = $layout;

                type Shape = [usize; $n];
                type Strides = [isize; $n];

                const RANK: usize = $n;
            }

            impl Shape for [usize; $n] {
                type Dim = Const<$n>;
            }

            impl Strides for [isize; $n] {
                type Dim = Const<$n>;
            }
        )*
    }
}

impl_dim!((0, 1, 2, 3, 4, 5, 6), (Dense, L::Uniform, L, L, L, L, L));
