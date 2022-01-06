use std::fmt::Debug;

/// Array dimension trait.
pub trait Dim: Copy + Debug + Default {
    /// Next larger dimension.
    type Larger: Dim;

    /// Next smaller dimension.
    type Smaller: Dim;

    /// Array shape type.
    type Shape: Shape<Dim = Self>;

    /// Array strides type.
    type Strides: Strides<Dim = Self>;

    /// Array rank.
    const RANK: usize;
}

/// Array shape trait.
pub trait Shape: AsMut<[usize]> + AsRef<[usize]> + Copy + Debug + Default {
    /// Array dimension type.
    type Dim: Dim<Shape = Self>;
}

/// Array strides trait.
pub trait Strides: AsMut<[isize]> + AsRef<[isize]> + Copy + Debug + Default {
    /// Array dimension type.
    type Dim: Dim<Strides = Self>;
}

/// Type-level constant.
#[derive(Clone, Copy, Debug, Default)]
pub struct Const<const N: usize>;

macro_rules! impl_dimension {
    ($($n:tt),*) => {
        $(
            impl Dim for Const<$n> {
                type Larger = Const<{ $n + ($n < 6) as usize }>;
                type Smaller = Const<{ $n - ($n > 0) as usize }>;

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

impl_dimension!(0, 1, 2, 3, 4, 5, 6);
