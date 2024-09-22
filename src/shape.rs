use std::fmt::Debug;

use crate::dim::{Const, Dim, Dims, Dyn, Strides};
use crate::layout::{Layout, Strided};

/// Array shape trait.
pub trait Shape: Copy + Debug + Default + Send + Sync {
    /// First dimension.
    type Head: Dim;

    /// Shape excluding the first dimension.
    type Tail: Shape;

    /// Shape with the reverse ordering of dimensions.
    type Reverse: Shape<Reverse = Self>;

    /// Append the dimension to the shape.
    type Append<D: Dim>: Shape<Reverse = <Self::Reverse as Shape>::Prepend<D>>;

    /// Prepend the dimension to the shape.
    type Prepend<D: Dim>: Shape<Reverse = <Self::Reverse as Shape>::Append<D>>;

    /// Merge each dimension pair, where constant size is preferred over dynamic.
    type Merge<S: Shape>: Shape;

    /// Select layout `L` or `Strided` for rank 0-1 or >1 respectively.
    type Layout<L: Layout>: Layout;

    /// Array dimensions type.
    type Dims: Dims;

    /// Array strides type.
    type Strides: Strides;

    /// Array rank, i.e. the number of dimensions.
    const RANK: usize;

    /// Returns the number of elements in each dimension.
    fn dims(self) -> Self::Dims;

    /// Creates an array shape with the given dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are not matching constant-sized dimensions.
    fn from_dims(dims: Self::Dims) -> Self;

    /// Returns the number of elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    fn dim(self, index: usize) -> usize {
        assert!(index < Self::RANK, "invalid dimension");

        self.dims()[index]
    }

    /// Returns `true` if the array contains no elements.
    fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements in the array.
    fn len(self) -> usize {
        self.dims()[..].iter().product()
    }

    /// Returns the array rank, i.e. the number of dimensions.
    fn rank(self) -> usize {
        Self::RANK
    }

    #[doc(hidden)]
    fn checked_len(self) -> Option<usize> {
        self.dims()[..].iter().try_fold(1usize, |acc, &x| acc.checked_mul(x))
    }

    #[doc(hidden)]
    fn prepend_dim<S: Shape>(self, size: usize) -> S {
        assert!(S::RANK == Self::RANK + 1, "invalid rank");

        let mut dims = S::Dims::default();

        dims[0] = size;
        dims[1..].copy_from_slice(&self.dims()[..]);

        S::from_dims(dims)
    }

    #[doc(hidden)]
    fn remove_dim<S: Shape>(self, index: usize) -> S {
        assert!(S::RANK + 1 == Self::RANK, "invalid rank");
        assert!(index < Self::RANK, "invalid dimension");

        let mut dims = S::Dims::default();

        dims[..index].copy_from_slice(&self.dims()[..index]);
        dims[index..].copy_from_slice(&self.dims()[index + 1..]);

        S::from_dims(dims)
    }

    #[doc(hidden)]
    fn resize_dim<S: Shape>(self, index: usize, new_size: usize) -> S {
        assert!(S::RANK == Self::RANK, "invalid rank");
        assert!(index < Self::RANK, "invalid dimension");

        let mut dims = S::Dims::default();

        dims[..].copy_from_slice(&self.dims()[..]);
        dims[index] = new_size;

        S::from_dims(dims)
    }

    #[doc(hidden)]
    fn reverse(self) -> Self::Reverse {
        let mut dims = <Self::Reverse as Shape>::Dims::default();

        dims[..].copy_from_slice(&self.dims()[..]);
        dims[..].reverse();

        Shape::from_dims(dims)
    }
}

/// Trait for array shape where all dimensions are constant-sized.
pub trait ConstShape: Shape {
    #[doc(hidden)]
    type Inner<T>;
}

/// Conversion trait into an array shape.
pub trait IntoShape {
    /// Which kind of array shape are we turning this into?
    type IntoShape: Shape;

    /// Creates array shape from a value.
    fn into_shape(self) -> Self::IntoShape;
}

/// Array shape type with dynamically-sized dimensions.
pub type Rank<const N: usize> = <[usize; N] as IntoShape>::IntoShape;

impl Shape for () {
    type Head = Dyn;
    type Tail = ();
    type Reverse = ();

    type Append<D: Dim> = (D,);
    type Prepend<D: Dim> = (D,);

    type Merge<S: Shape> = S;
    type Layout<L: Layout> = L;

    type Dims = [usize; 0];
    type Strides = [isize; 0];

    const RANK: usize = 0;

    fn dims(self) -> [usize; 0] {
        []
    }

    fn from_dims(_: [usize; 0]) -> Self {}
}

impl<X: Dim> Shape for (X,) {
    type Head = X;
    type Tail = ();
    type Reverse = (X,);

    type Append<D: Dim> = (X, D);
    type Prepend<D: Dim> = (D, X);

    type Merge<S: Shape> = <S::Tail as Shape>::Prepend<X::Merge<S::Head>>;
    type Layout<L: Layout> = L;

    type Dims = [usize; 1];
    type Strides = [isize; 1];

    const RANK: usize = 1;

    fn dims(self) -> [usize; 1] {
        [self.0.size()]
    }

    fn from_dims(dims: [usize; 1]) -> Self {
        (X::from_size(dims[0]),)
    }
}

macro_rules! impl_shape {
    ($n:tt, ($($jk:tt),*), ($($yz:tt),*), $append:tt, $prepend:tt) => {
        impl<X: Dim, $($yz: Dim,)+> Shape for (X, $($yz,)+) {
            type Head = X;
            type Tail = ($($yz,)+);
            type Reverse = <<Self::Tail as Shape>::Reverse as Shape>::Append<X>;

            type Append<D: Dim> = $append;
            type Prepend<D: Dim> = $prepend;

            type Merge<S: Shape> =
                <<Self::Tail as Shape>::Merge<S::Tail> as Shape>::Prepend<X::Merge<S::Head>>;
            type Layout<L: Layout> = Strided;

            type Dims = [usize; $n];
            type Strides = [isize; $n];

            const RANK: usize = $n;

            fn dims(self) -> [usize; $n] {
                [self.0.size() $(,self.$jk.size())+]
            }

            fn from_dims(dims: [usize; $n]) -> Self {
                (X::from_size(dims[0]) $(,$yz::from_size(dims[$jk]))+)
            }
        }
    };
}

impl_shape!(2, (1), (Y), (X, Y, D), (D, X, Y));
impl_shape!(3, (1, 2), (Y, Z), (X, Y, Z, D), (D, X, Y, Z));
impl_shape!(4, (1, 2, 3), (Y, Z, W), (X, Y, Z, W, D), (D, X, Y, Z, W));
impl_shape!(5, (1, 2, 3, 4), (Y, Z, W, U), (X, Y, Z, W, U, D), (D, X, Y, Z, W, U));
impl_shape!(6, (1, 2, 3, 4, 5), (Y, Z, W, U, V), (Y, Z, W, U, V, D), (D, X, Y, Z, W, U));

macro_rules! impl_const_shape {
    (($($xyz:tt),*), $inner:ty) => {
        impl<$(const $xyz: usize),*> ConstShape for ($(Const<$xyz>,)*) {
            type Inner<T> = $inner;
        }
    };
}

impl_const_shape!((), T);
impl_const_shape!((X), [T; X]);
impl_const_shape!((X, Y), [[T; Y]; X]);
impl_const_shape!((X, Y, Z), [[[T; Z]; Y]; X]);
impl_const_shape!((X, Y, Z, W), [[[[T; W]; Z]; Y]; X]);
impl_const_shape!((X, Y, Z, W, U), [[[[[T; U]; W]; Z]; Y]; X]);
impl_const_shape!((X, Y, Z, W, U, V), [[[[[[T; V]; U]; W]; Z]; Y]; X]);

impl<S: Shape> IntoShape for S {
    type IntoShape = S;

    fn into_shape(self) -> S {
        self
    }
}

macro_rules! impl_into_shape {
    ($n:tt, $shape:ty) => {
        impl IntoShape for [usize; $n] {
            type IntoShape = $shape;

            fn into_shape(self) -> Self::IntoShape {
                Self::IntoShape::from_dims(self)
            }
        }
    };
}

impl_into_shape!(0, ());
impl_into_shape!(1, (Dyn,));
impl_into_shape!(2, (Dyn, Dyn));
impl_into_shape!(3, (Dyn, Dyn, Dyn));
impl_into_shape!(4, (Dyn, Dyn, Dyn, Dyn));
impl_into_shape!(5, (Dyn, Dyn, Dyn, Dyn, Dyn));
impl_into_shape!(6, (Dyn, Dyn, Dyn, Dyn, Dyn, Dyn));
