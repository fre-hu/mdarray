#[cfg(feature = "nightly")]
use std::alloc::Allocator;
use std::cmp::Ordering;
use std::mem;

use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::array::{Array, GridArray, SpanArray};
use crate::buffer::{Buffer, BufferMut};
use crate::dim::{Const, Dim};
use crate::format::Format;
use crate::layout::DenseLayout;

/// Fill value to be used as scalar operand for array operators.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Fill<T: Copy> {
    /// Fill value.
    pub value: T,
}

/// Range constructed from a unit spaced range with the given step size.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct StepRange<R, S> {
    /// Unit spaced range.
    pub range: R,

    /// Step size.
    pub step: S,
}

/// Returns a fill value to be used as scalar operand for array operators.
///
/// If the type does not implement the `Copy` trait, a reference must be passed as input.
#[must_use]
pub fn fill<T: Copy>(value: T) -> Fill<T> {
    Fill { value }
}

/// Returns a range with the given step size from a unit spaced range.
///
/// If the step size is negative, the result is the reverse of the corresponding range
/// with step size as the absolute value of the given step size.
///
/// For example, `step(0..10, 2)` gives the values `0, 2, 4, 6, 8` and `step(0..10, -2)`
/// gives the values `8, 6, 4, 2, 0`.
#[must_use]
pub fn step<R, S>(range: R, step: S) -> StepRange<R, S> {
    StepRange { range, step }
}

impl<T: Eq, B: Buffer<Item = T> + ?Sized> Eq for Array<B> where Self: PartialEq {}

impl<T: Ord, B: Buffer<Item = T, Dim = Const<1>> + ?Sized> Ord for Array<B> {
    fn cmp(&self, other: &Self) -> Ordering {
        if B::Format::IS_UNIFORM && B::Format::IS_UNIT_STRIDED {
            self.as_span().reformat().as_slice().cmp(other.as_span().reformat().as_slice())
        } else {
            self.as_span().flatten().iter().cmp(other.as_span().flatten().iter())
        }
    }
}

impl<B: Buffer<Dim = Const<1>> + ?Sized, C: Buffer<Dim = Const<1>> + ?Sized> PartialOrd<Array<C>>
    for Array<B>
where
    B::Item: PartialOrd<C::Item>,
{
    fn partial_cmp(&self, other: &Array<C>) -> Option<Ordering> {
        self.as_span().flatten().iter().partial_cmp(other.as_span().flatten().iter())
    }
}

impl<D: Dim, B: Buffer<Dim = D> + ?Sized, C: Buffer<Dim = D> + ?Sized> PartialEq<Array<C>>
    for Array<B>
where
    B::Item: PartialEq<C::Item>,
{
    fn eq(&self, other: &Array<C>) -> bool {
        if B::Format::IS_UNIFORM && C::Format::IS_UNIFORM {
            if self.as_span().shape()[..] == other.as_span().shape()[..] {
                if B::Format::IS_UNIT_STRIDED && C::Format::IS_UNIT_STRIDED {
                    self.as_span().reformat().as_slice().eq(other.as_span().reformat().as_slice())
                } else {
                    self.as_span().flatten().iter().eq(other.as_span().flatten().iter())
                }
            } else {
                false
            }
        } else {
            self.as_span().outer_iter().eq(other.as_span().outer_iter())
        }
    }
}

macro_rules! impl_binary_op {
    ($trt:tt, $fn:tt) => {
        impl<T, D: Dim, B: Buffer<Dim = D> + ?Sized, C: Buffer<Dim = D> + ?Sized> $trt<&Array<C>>
            for &Array<B>
        where
            for<'a, 'b> &'a B::Item: $trt<&'b C::Item, Output = T>,
        {
            type Output = GridArray<T, D>;

            fn $fn(self, rhs: &Array<C>) -> Self::Output {
                let mut vec = Vec::with_capacity(self.as_span().len());

                unsafe {
                    from_binary_op(&mut vec, self.as_span(), rhs.as_span(), &|x, y| x.$fn(y));

                    GridArray::from_parts(vec, DenseLayout::new(self.as_span().shape()))
                }
            }
        }

        impl<T: Copy, U, B: Buffer + ?Sized> $trt<Fill<T>> for &Array<B>
        where
            for<'a> &'a B::Item: $trt<T, Output = U>,
        {
            type Output = GridArray<U, B::Dim>;

            fn $fn(self, rhs: Fill<T>) -> Self::Output {
                let mut vec = Vec::with_capacity(self.as_span().len());

                unsafe {
                    from_unary_op(&mut vec, self.as_span(), &|x| x.$fn(rhs.value));

                    GridArray::from_parts(vec, DenseLayout::new(self.as_span().shape()))
                }
            }
        }

        impl<T: Default, D: Dim, B: Buffer<Dim = D> + ?Sized, A: Allocator> $trt<GridArray<T, D, A>>
            for &Array<B>
        where
            for<'a> &'a B::Item: $trt<T, Output = T>,
        {
            type Output = GridArray<T, D, A>;

            fn $fn(self, mut rhs: GridArray<T, D, A>) -> Self::Output {
                map_binary_op(&mut rhs, self.as_span(), &|(x, y)| *x = y.$fn(mem::take(x)));

                rhs
            }
        }

        impl<T: Copy, U, B: Buffer + ?Sized> $trt<&Array<B>> for Fill<T>
        where
            for<'a> T: $trt<&'a B::Item, Output = U>,
        {
            type Output = GridArray<U, B::Dim>;

            fn $fn(self, rhs: &Array<B>) -> Self::Output {
                let mut vec = Vec::with_capacity(rhs.as_span().len());

                unsafe {
                    from_unary_op(&mut vec, rhs.as_span(), &|x| self.value.$fn(x));

                    GridArray::from_parts(vec, DenseLayout::new(rhs.as_span().shape()))
                }
            }
        }

        impl<T: Copy, U: Default, D: Dim, A: Allocator> $trt<GridArray<U, D, A>> for Fill<T>
        where
            T: $trt<U, Output = U>,
        {
            type Output = GridArray<U, D, A>;

            fn $fn(self, mut rhs: GridArray<U, D, A>) -> Self::Output {
                map_unary_op(&mut rhs, &|x| *x = self.value.$fn(mem::take(x)));

                rhs
            }
        }

        impl<T: Default, D: Dim, B: Buffer<Dim = D> + ?Sized, A: Allocator> $trt<&Array<B>>
            for GridArray<T, D, A>
        where
            for<'a> T: $trt<&'a B::Item, Output = T>,
        {
            type Output = Self;

            fn $fn(mut self, rhs: &Array<B>) -> Self {
                map_binary_op(&mut self, rhs.as_span(), &|(x, y)| *x = mem::take(x).$fn(y));

                self
            }
        }

        impl<T: Default, U: Copy, D: Dim, A: Allocator> $trt<Fill<U>> for GridArray<T, D, A>
        where
            T: $trt<U, Output = T>,
        {
            type Output = Self;

            fn $fn(mut self, rhs: Fill<U>) -> Self {
                map_unary_op(&mut self, &|x| *x = mem::take(x).$fn(rhs.value));

                self
            }
        }
    };
}

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);
impl_binary_op!(Div, div);
impl_binary_op!(Rem, rem);
impl_binary_op!(BitAnd, bitand);
impl_binary_op!(BitOr, bitor);
impl_binary_op!(BitXor, bitxor);
impl_binary_op!(Shl, shl);
impl_binary_op!(Shr, shr);

macro_rules! impl_op_assign {
    ($trt:tt, $fn:tt) => {
        impl<D: Dim, B: BufferMut<Dim = D> + ?Sized, C: Buffer<Dim = D> + ?Sized> $trt<&Array<C>>
            for Array<B>
        where
            for<'a> B::Item: $trt<&'a C::Item>,
        {
            fn $fn(&mut self, rhs: &Array<C>) {
                map_binary_op(self.as_mut_span(), rhs.as_span(), &|(x, y)| x.$fn(y));
            }
        }

        impl<T: Copy, B: BufferMut + ?Sized> $trt<Fill<T>> for Array<B>
        where
            for<'a> B::Item: $trt<T>,
        {
            fn $fn(&mut self, rhs: Fill<T>) {
                map_unary_op(self.as_mut_span(), &|x| x.$fn(rhs.value));
            }
        }
    };
}

impl_op_assign!(AddAssign, add_assign);
impl_op_assign!(SubAssign, sub_assign);
impl_op_assign!(MulAssign, mul_assign);
impl_op_assign!(DivAssign, div_assign);
impl_op_assign!(RemAssign, rem_assign);
impl_op_assign!(BitAndAssign, bitand_assign);
impl_op_assign!(BitOrAssign, bitor_assign);
impl_op_assign!(BitXorAssign, bitxor_assign);
impl_op_assign!(ShlAssign, shl_assign);
impl_op_assign!(ShrAssign, shr_assign);

macro_rules! impl_unary_op {
    ($trt:tt, $fn:tt) => {
        impl<B: Buffer + ?Sized> $trt for &Array<B>
        where
            for<'a> &'a B::Item: $trt<Output = B::Item>,
        {
            type Output = GridArray<B::Item, B::Dim>;

            fn $fn(self) -> Self::Output {
                let mut vec = Vec::with_capacity(self.as_span().len());

                unsafe {
                    from_unary_op(&mut vec, self.as_span(), &|x| x.$fn());

                    GridArray::from_parts(vec, DenseLayout::new(self.as_span().shape()))
                }
            }
        }

        impl<T: Default, D: Dim, A: Allocator> $trt for GridArray<T, D, A>
        where
            T: $trt<Output = T>,
        {
            type Output = Self;

            fn $fn(mut self) -> Self {
                map_unary_op(&mut self, &|x| *x = mem::take(x).$fn());

                self
            }
        }
    };
}

impl_unary_op!(Neg, neg);
impl_unary_op!(Not, not);

unsafe fn from_binary_op<T, F: Format, G: Format, U, V, D: Dim>(
    vec: &mut Vec<V>,
    lhs: &SpanArray<T, D, F>,
    rhs: &SpanArray<U, D, G>,
    f: &impl Fn(&T, &U) -> V,
) {
    if F::IS_UNIFORM && G::IS_UNIFORM {
        assert!(lhs.shape()[..] == rhs.shape()[..], "shape mismatch");

        for (x, y) in lhs.flatten().iter().zip(rhs.flatten().iter()) {
            vec.as_mut_ptr().add(vec.len()).write(f(x, y));
            vec.set_len(vec.len() + 1);
        }
    } else {
        assert!(lhs.size(D::RANK - 1) == rhs.size(D::RANK - 1), "shape mismatch");

        for (x, y) in lhs.outer_iter().zip(rhs.outer_iter()) {
            from_binary_op(vec, &x, &y, f);
        }
    }
}

unsafe fn from_unary_op<T, F: Format, U>(
    vec: &mut Vec<U>,
    other: &SpanArray<T, impl Dim, F>,
    f: &impl Fn(&T) -> U,
) {
    if F::IS_UNIFORM {
        for x in other.flatten().iter() {
            vec.as_mut_ptr().add(vec.len()).write(f(x));
            vec.set_len(vec.len() + 1);
        }
    } else {
        for x in other.outer_iter() {
            from_unary_op(vec, &x, f);
        }
    }
}

fn map_binary_op<T, F: Format, G: Format, U, D: Dim>(
    this: &mut SpanArray<T, D, F>,
    other: &SpanArray<U, D, G>,
    f: &impl Fn((&mut T, &U)),
) {
    if F::IS_UNIFORM && G::IS_UNIFORM {
        assert!(this.shape()[..] == other.shape()[..], "shape mismatch");

        this.flatten_mut().iter_mut().zip(other.flatten().iter()).for_each(f);
    } else {
        assert!(this.size(D::RANK - 1) == other.size(D::RANK - 1), "shape mismatch");

        for (mut x, y) in this.outer_iter_mut().zip(other.outer_iter()) {
            map_binary_op(&mut x, &y, f);
        }
    }
}

fn map_unary_op<T, F: Format>(this: &mut SpanArray<T, impl Dim, F>, f: &impl Fn(&mut T)) {
    if F::IS_UNIFORM {
        this.flatten_mut().iter_mut().for_each(f);
    } else {
        for mut x in this.outer_iter_mut() {
            map_unary_op(&mut x, f);
        }
    }
}
