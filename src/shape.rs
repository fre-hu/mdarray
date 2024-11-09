use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::slice;

use crate::dim::{Const, Dim, Dims, Dyn};
use crate::layout::{Layout, Strided};

/// Array shape trait.
pub trait Shape: Clone + Debug + Default + Eq + Hash + Send + Sync {
    /// First dimension.
    type Head: Dim;

    /// Shape excluding the first dimension.
    type Tail: Shape;

    /// Shape with the reverse ordering of dimensions.
    type Reverse: Shape<Reverse = Self>;

    /// Prepend the dimension to the shape.
    type Prepend<D: Dim>: Shape;

    /// Concatenate the other shape to the shape.
    type Concat<S: Shape>: Shape;

    /// Merge each dimension pair, where constant size is preferred over dynamic.
    /// The result has dynamic rank if at least one of the inputs has dynamic rank.
    type Merge<S: Shape>: Shape;

    /// Select layout `L` for rank 0, or `Strided` for rank >0 or dynamic.
    type Layout<L: Layout>: Layout;

    #[doc(hidden)]
    type Dims<T: Copy + Debug + Default + Eq + Hash + Send + Sync>: Dims<T>;

    /// Array rank if known statically, or `None` if dynamic.
    const RANK: Option<usize>;

    /// Returns the number of elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    fn dim(&self, index: usize) -> usize {
        assert!(index < self.rank(), "invalid dimension");

        self.with_dims(|dims| dims[index])
    }

    /// Creates an array shape with the given dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are not matching static rank or constant-sized dimensions.
    fn from_dims(dims: &[usize]) -> Self {
        let mut shape = Self::new(dims.len());

        shape.with_mut_dims(|dst| dst.copy_from_slice(dims));
        shape
    }

    /// Returns `true` if the array contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements in the array.
    fn len(&self) -> usize {
        self.with_dims(|dims| dims.iter().product())
    }

    /// Returns the array rank, i.e. the number of dimensions.
    fn rank(&self) -> usize {
        self.with_dims(|dims| dims.len())
    }

    #[doc(hidden)]
    fn new(rank: usize) -> Self;

    #[doc(hidden)]
    fn with_dims<T, F: FnMut(&[usize]) -> T>(&self, f: F) -> T;

    #[doc(hidden)]
    fn with_mut_dims<T, F: FnMut(&mut [usize]) -> T>(&mut self, f: F) -> T;

    #[doc(hidden)]
    fn checked_len(&self) -> Option<usize> {
        self.with_dims(|dims| dims.iter().try_fold(1usize, |acc, &x| acc.checked_mul(x)))
    }

    #[doc(hidden)]
    fn prepend_dim<S: Shape>(&self, size: usize) -> S {
        let mut shape = S::new(self.rank() + 1);

        shape.with_mut_dims(|dims| {
            dims[0] = size;
            self.with_dims(|src| dims[1..].copy_from_slice(src));
        });

        shape
    }

    #[doc(hidden)]
    fn remove_dim<S: Shape>(&self, index: usize) -> S {
        assert!(index < self.rank(), "invalid dimension");

        let mut shape = S::new(self.rank() - 1);

        shape.with_mut_dims(|dims| {
            self.with_dims(|src| {
                dims[..index].copy_from_slice(&src[..index]);
                dims[index..].copy_from_slice(&src[index + 1..]);
            });
        });

        shape
    }

    #[doc(hidden)]
    fn resize_dim<S: Shape>(&self, index: usize, new_size: usize) -> S {
        assert!(index < self.rank(), "invalid dimension");

        let mut shape = S::new(self.rank());

        shape.with_mut_dims(|dims| {
            self.with_dims(|src| dims[..].copy_from_slice(src));
            dims[index] = new_size;
        });

        shape
    }

    #[doc(hidden)]
    fn reverse(&self) -> Self::Reverse {
        let mut shape = Self::Reverse::new(self.rank());

        shape.with_mut_dims(|dims| {
            self.with_dims(|src| dims.copy_from_slice(src));
            dims.reverse();
        });

        shape
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

/// Array shape type with dynamic rank.
///
/// If the rank is 0 or 1, no heap allocation is necessary. The default value
/// will have rank 1 and contain no elements.
pub enum DynRank {
    /// Shape variant with dynamic rank.
    Dyn(Box<[usize]>),
    /// Shape variant with rank 1.
    One(usize),
}

/// Array shape type with dynamically-sized dimensions.
pub type Rank<const N: usize> = <[usize; N] as IntoShape>::IntoShape;

impl DynRank {
    /// Returns the number of elements in each dimension.
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::Dyn(dims) => dims,
            Self::One(size) => slice::from_ref(size),
        }
    }
}

impl Clone for DynRank {
    fn clone(&self) -> Self {
        match self {
            DynRank::One(dim) => DynRank::One(*dim),
            DynRank::Dyn(dims) => {
                if dims.len() == 1 {
                    DynRank::One(dims[0])
                } else {
                    DynRank::Dyn(dims.clone())
                }
            }
        }
    }

    fn clone_from(&mut self, source: &Self) {
        if let DynRank::Dyn(dims) = self {
            if let DynRank::Dyn(src) = source {
                if dims.len() == src.len() {
                    dims.clone_from_slice(src);

                    return;
                }
            }
        }

        *self = source.clone();
    }
}

impl Debug for DynRank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.with_dims(|dims| f.debug_tuple("DynRank").field(&dims).finish())
    }
}

impl Default for DynRank {
    fn default() -> Self {
        Self::One(0)
    }
}

impl Eq for DynRank {}

impl Hash for DynRank {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.with_dims(|dims| dims.hash(state))
    }
}

impl PartialEq for DynRank {
    fn eq(&self, other: &Self) -> bool {
        self.with_dims(|dims| other.with_dims(|other| dims == other))
    }
}

impl Shape for DynRank {
    type Head = Dyn;
    type Tail = Self;
    type Reverse = Self;

    type Prepend<D: Dim> = Self;
    type Concat<S: Shape> = Self;
    type Merge<S: Shape> = Self;

    type Layout<L: Layout> = Strided;
    type Dims<T: Copy + Debug + Default + Eq + Hash + Send + Sync> = Box<[T]>;

    const RANK: Option<usize> = None;

    fn new(rank: usize) -> Self {
        if rank == 1 {
            DynRank::One(0)
        } else {
            DynRank::Dyn(Dims::new(rank))
        }
    }

    fn with_dims<T, F: FnMut(&[usize]) -> T>(&self, mut f: F) -> T {
        let dims = match self {
            Self::Dyn(dims) => dims,
            Self::One(size) => slice::from_ref(size),
        };

        f(dims)
    }

    fn with_mut_dims<T, F: FnMut(&mut [usize]) -> T>(&mut self, mut f: F) -> T {
        let dims = match self {
            Self::Dyn(dims) => dims,
            Self::One(size) => slice::from_mut(size),
        };

        f(dims)
    }
}

impl Shape for () {
    type Head = Dyn;
    type Tail = ();
    type Reverse = ();

    type Prepend<D: Dim> = (D,);
    type Concat<S: Shape> = S;
    type Merge<S: Shape> = S;

    type Layout<L: Layout> = L;
    type Dims<T: Copy + Debug + Default + Eq + Hash + Send + Sync> = [T; 0];

    const RANK: Option<usize> = Some(0);

    fn new(rank: usize) {
        assert!(rank == 0, "invalid rank");
    }

    fn with_dims<T, F: FnMut(&[usize]) -> T>(&self, mut f: F) -> T {
        f(&[])
    }

    fn with_mut_dims<T, F: FnMut(&mut [usize]) -> T>(&mut self, mut f: F) -> T {
        f(&mut [])
    }
}

impl<X: Dim> Shape for (X,) {
    type Head = X;
    type Tail = ();
    type Reverse = (X,);

    type Prepend<D: Dim> = (D, X);
    type Concat<S: Shape> = S::Prepend<X>;
    type Merge<S: Shape> = <S::Tail as Shape>::Prepend<X::Merge<S::Head>>;

    type Layout<L: Layout> = Strided;
    type Dims<T: Copy + Debug + Default + Eq + Hash + Send + Sync> = [T; 1];

    const RANK: Option<usize> = Some(1);

    fn new(rank: usize) -> Self {
        assert!(rank == 1, "invalid rank");

        Self::default()
    }

    fn with_dims<T, F: FnMut(&[usize]) -> T>(&self, mut f: F) -> T {
        f(&[self.0.size()])
    }

    fn with_mut_dims<T, F: FnMut(&mut [usize]) -> T>(&mut self, mut f: F) -> T {
        let mut dims = [self.0.size()];
        let value = f(&mut dims);

        *self = (X::from_size(dims[0]),);

        value
    }
}

macro_rules! impl_shape {
    ($n:tt, ($($jk:tt),*), ($($yz:tt),*), $prepend:tt) => {
        impl<X: Dim, $($yz: Dim,)+> Shape for (X, $($yz,)+) {
            type Head = X;
            type Tail = ($($yz,)+);
            type Reverse = <<Self::Tail as Shape>::Reverse as Shape>::Concat<(X,)>;

            type Prepend<D: Dim> = $prepend;
            type Concat<S: Shape> = <<Self::Tail as Shape>::Concat<S> as Shape>::Prepend<X>;
            type Merge<S: Shape> =
                <<Self::Tail as Shape>::Merge<S::Tail> as Shape>::Prepend<X::Merge<S::Head>>;

            type Layout<L: Layout> = Strided;
            type Dims<T: Copy + Debug + Default + Eq + Hash + Send + Sync> = [T; $n];

            const RANK: Option<usize> = Some($n);

            fn new(rank: usize) -> Self {
                assert!(rank == $n, "invalid rank");

                Self::default()
            }

            fn with_dims<T, F: FnMut(&[usize]) -> T>(&self, mut f: F) -> T {
                f(&[self.0.size() $(,self.$jk.size())+])
            }

            fn with_mut_dims<T, F: FnMut(&mut [usize]) -> T>(&mut self, mut f: F) -> T {
                let mut dims = [self.0.size() $(,self.$jk.size())+];
                let value = f(&mut dims);

                *self = (X::from_size(dims[0]) $(,$yz::from_size(dims[$jk]))+);

                value
            }
        }
    };
}

impl_shape!(2, (1), (Y), (D, X, Y));
impl_shape!(3, (1, 2), (Y, Z), (D, X, Y, Z));
impl_shape!(4, (1, 2, 3), (Y, Z, W), (D, X, Y, Z, W));
impl_shape!(5, (1, 2, 3, 4), (Y, Z, W, U), (D, X, Y, Z, W, U));
impl_shape!(6, (1, 2, 3, 4, 5), (Y, Z, W, U, V), DynRank);

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

impl IntoShape for &[usize] {
    type IntoShape = DynRank;

    fn into_shape(self) -> DynRank {
        DynRank::from_dims(self)
    }
}

impl IntoShape for Box<[usize]> {
    type IntoShape = DynRank;

    fn into_shape(self) -> DynRank {
        DynRank::Dyn(self)
    }
}

impl IntoShape for Vec<usize> {
    type IntoShape = DynRank;

    fn into_shape(self) -> DynRank {
        DynRank::Dyn(self.into())
    }
}

macro_rules! impl_into_shape {
    ($n:tt, $shape:ty) => {
        impl IntoShape for [usize; $n] {
            type IntoShape = $shape;

            fn into_shape(self) -> Self::IntoShape {
                Self::IntoShape::from_dims(&self)
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
