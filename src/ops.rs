#[cfg(feature = "nightly")]
use std::alloc::Allocator;
use std::cmp::Ordering;

use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::array::{Array, GridArray, ViewArray};
use crate::buffer::{Buffer, BufferMut};
use crate::dim::{Const, Dim};
use crate::expr::Producer;
use crate::expression::Expression;
use crate::layout::Layout;
use crate::traits::{Apply, IntoExpression};

/// Range constructed from a unit spaced range with the given step size.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct StepRange<R, S> {
    /// Unit spaced range.
    pub range: R,

    /// Step size.
    pub step: S,
}

/// Creates a range with the given step size from a unit spaced range.
///
/// If the step size is negative, the result is obtained by reversing the input range
/// and stepping by the absolute value of the step size.
///
/// # Examples
///
/// ```
/// use mdarray::{step, view, View};
///
/// let v = view![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
///
/// assert_eq!(v.view(step(0..10, 2)).to_vec(), [0, 2, 4, 6, 8]);
/// assert_eq!(v.view(step(0..10, -2)).to_vec(), [9, 7, 5, 3, 1]);
/// ```
pub fn step<R, S>(range: R, step: S) -> StepRange<R, S> {
    StepRange { range, step }
}

impl<B: Buffer<Item: Eq> + ?Sized> Eq for Array<B> where Self: PartialEq {}

impl<B: Buffer<Item: Ord, Dim = Const<1>> + ?Sized> Ord for Array<B> {
    fn cmp(&self, other: &Self) -> Ordering {
        if B::Layout::IS_UNIT_STRIDED {
            self.as_span().remap().as_slice().cmp(other.as_span().remap().as_slice())
        } else {
            self.as_span().iter().cmp(other)
        }
    }
}

impl<T: PartialOrd, B, C> PartialOrd<Array<C>> for Array<B>
where
    B: Buffer<Item = T, Dim = Const<1>> + ?Sized,
    C: Buffer<Item = T, Dim = Const<1>> + ?Sized,
{
    fn partial_cmp(&self, other: &Array<C>) -> Option<Ordering> {
        if B::Layout::IS_UNIT_STRIDED {
            self.as_span().remap().as_slice().partial_cmp(other.as_span().remap().as_slice())
        } else {
            self.as_span().iter().partial_cmp(other)
        }
    }
}

impl<D: Dim, B: Buffer<Dim = D> + ?Sized, C: Buffer<Dim = D> + ?Sized> PartialEq<Array<C>>
    for Array<B>
where
    B::Item: PartialEq<C::Item>,
{
    fn eq(&self, other: &Array<C>) -> bool {
        if self.as_span().shape()[..] == other.as_span().shape()[..] {
            if B::Layout::IS_UNIFORM && C::Layout::IS_UNIFORM {
                if B::Layout::IS_UNIT_STRIDED && C::Layout::IS_UNIT_STRIDED {
                    self.as_span().remap().as_slice().eq(other.as_span().remap().as_slice())
                } else {
                    self.as_span().iter().eq(other)
                }
            } else {
                self.as_span().outer_expr().into_iter().eq(other.as_span().outer_expr())
            }
        } else {
            false
        }
    }
}

macro_rules! impl_binary_op {
    ($trt:tt, $fn:tt) => {
        impl<'a, T, B: Buffer + ?Sized, I: Apply<T>> $trt<I> for &'a Array<B>
        where
            for<'b> &'b B::Item: $trt<I::Item, Output = T>,
        {
            type Output = I::ZippedWith<Self, impl FnMut(I::Item, &'a B::Item) -> T>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |x, y| y.$fn(x))
            }
        }

        impl<T, P: Producer, I: Apply<T>> $trt<I> for Expression<P>
        where
            P::Item: $trt<I::Item, Output = T>,
        {
            type Output = I::ZippedWith<Self, impl FnMut(I::Item, P::Item) -> T>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |x, y| y.$fn(x))
            }
        }

        impl<T, D: Dim, I: IntoExpression, A: Allocator> $trt<I> for GridArray<T, D, A>
        where
            T: $trt<I::Item, Output = T>,
        {
            type Output = Self;

            fn $fn(self, rhs: I) -> Self {
                self.zip_with(rhs, |x, y| x.$fn(y))
            }
        }

        impl<'a, T, U, D: Dim, I: Apply<U>, L: Layout> $trt<I> for ViewArray<'a, T, D, L>
        where
            for<'b> &'b T: $trt<I::Item, Output = U>,
        {
            type Output = I::ZippedWith<Self, impl FnMut(I::Item, &'a T) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |x, y| y.$fn(x))
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
        impl<B: BufferMut + ?Sized, I: IntoExpression> $trt<I> for Array<B>
        where
            B::Item: $trt<I::Item>,
        {
            fn $fn(&mut self, rhs: I) {
                self.as_mut_span().expr_mut().zip(rhs).for_each(|(x, y)| x.$fn(y));
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
        impl<T, B: Buffer + ?Sized> $trt for &Array<B>
        where
            for<'a> &'a B::Item: $trt<Output = T>,
        {
            type Output = Expression<impl Producer<Item = T, Dim = B::Dim>>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<T, P: Producer> $trt for Expression<P>
        where
            P::Item: $trt<Output = T>,
        {
            type Output = Expression<impl Producer<Item = T, Dim = P::Dim>>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<T, D: Dim, A: Allocator> $trt for GridArray<T, D, A>
        where
            T: $trt<Output = T>,
        {
            type Output = Self;

            fn $fn(self) -> Self {
                self.apply(|x| x.$fn())
            }
        }

        impl<'a, T, U, D: Dim, L: Layout> $trt for ViewArray<'a, T, D, L>
        where
            for<'b> &'b T: $trt<Output = U>,
        {
            type Output = Expression<impl Producer<Item = U, Dim = D>>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }
    };
}

impl_unary_op!(Neg, neg);
impl_unary_op!(Not, not);
