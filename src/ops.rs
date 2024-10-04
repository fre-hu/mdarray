#[cfg(feature = "nightly")]
use std::alloc::Allocator;

use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::array::Array;
use crate::buffer::Buffer;
use crate::expr::{Fill, FillWith, FromElem, FromFn, IntoExpr, Map};
use crate::expression::Expression;
use crate::layout::Layout;
use crate::shape::{ConstShape, Shape};
use crate::slice::Slice;
use crate::tensor::Tensor;
use crate::traits::{Apply, IntoExpression};
use crate::view::{View, ViewMut};

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
/// use mdarray::{step, view};
///
/// let v = view![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
///
/// assert_eq!(v.view(step(0..10, 2)).to_vec(), [0, 2, 4, 6, 8]);
/// assert_eq!(v.view(step(0..10, -2)).to_vec(), [9, 7, 5, 3, 1]);
/// ```
pub fn step<R, S>(range: R, step: S) -> StepRange<R, S> {
    StepRange { range, step }
}

impl<T: Eq, S: ConstShape> Eq for Array<T, S> {}
impl<T: Eq, S: Shape, L: Layout> Eq for Slice<T, S, L> {}
impl<T: Eq, S: Shape, A: Allocator> Eq for Tensor<T, S, A> {}
impl<T: Eq, S: Shape, L: Layout> Eq for View<'_, T, S, L> {}
impl<T: Eq, S: Shape, L: Layout> Eq for ViewMut<'_, T, S, L> {}

impl<U, V, S: ConstShape, T: Shape, L: Layout, I: ?Sized> PartialEq<I> for Array<U, S>
where
    for<'a> &'a I: IntoExpression<IntoExpr = View<'a, V, T, L>>,
    U: PartialEq<V>,
{
    fn eq(&self, other: &I) -> bool {
        (**self).eq(other)
    }
}

impl<U, V, S: Shape, T: Shape, L: Layout, M: Layout, I: ?Sized> PartialEq<I> for Slice<U, S, L>
where
    for<'a> &'a I: IntoExpression<IntoExpr = View<'a, V, T, M>>,
    U: PartialEq<V>,
{
    fn eq(&self, other: &I) -> bool {
        let other = other.into_expr();

        if self.dims()[..] == other.dims()[..] {
            // Avoid very long compile times for release build with MIR inlining,
            // by avoiding recursion until types are known.
            //
            // This is a workaround until const if is available, see #3582 and #122301.

            fn compare_inner<U, V, S: Shape, T: Shape, L: Layout, M: Layout>(
                this: &Slice<U, S, L>,
                other: &Slice<V, T, M>,
            ) -> bool
            where
                U: PartialEq<V>,
            {
                if L::IS_DENSE && M::IS_DENSE {
                    this.remap()[..].eq(&other.remap()[..])
                } else {
                    this.iter().eq(other)
                }
            }

            fn compare_outer<U, V, S: Shape, T: Shape, L: Layout, M: Layout>(
                this: &Slice<U, S, L>,
                other: &Slice<V, T, M>,
            ) -> bool
            where
                U: PartialEq<V>,
            {
                this.outer_expr().into_iter().eq(other.outer_expr())
            }

            let f = const {
                if S::RANK < 2 || (L::IS_DENSE && M::IS_DENSE) {
                    compare_inner
                } else {
                    compare_outer
                }
            };

            f(self, &other)
        } else {
            false
        }
    }
}

impl<U, V, S: Shape, T: Shape, L: Layout, A: Allocator, I: ?Sized> PartialEq<I> for Tensor<U, S, A>
where
    for<'a> &'a I: IntoExpression<IntoExpr = View<'a, V, T, L>>,
    U: PartialEq<V>,
{
    fn eq(&self, other: &I) -> bool {
        (**self).eq(other)
    }
}

impl<U, V, S: Shape, T: Shape, L: Layout, M: Layout, I: ?Sized> PartialEq<I> for View<'_, U, S, L>
where
    for<'a> &'a I: IntoExpression<IntoExpr = View<'a, V, T, M>>,
    U: PartialEq<V>,
{
    fn eq(&self, other: &I) -> bool {
        (**self).eq(other)
    }
}

impl<U, V, S: Shape, T: Shape, L: Layout, M: Layout, I: ?Sized> PartialEq<I>
    for ViewMut<'_, U, S, L>
where
    for<'a> &'a I: IntoExpression<IntoExpr = View<'a, V, T, M>>,
    U: PartialEq<V>,
{
    fn eq(&self, other: &I) -> bool {
        (**self).eq(other)
    }
}

macro_rules! impl_binary_op {
    ($trt:tt, $fn:tt) => {
        impl<'a, T, U, S: ConstShape, I: Apply<U>> $trt<I> for &'a Array<T, S>
        where
            &'a T: $trt<I::Item, Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = I::ZippedWith<Self, fn((I::Item, &'a T)) -> U>;

            #[cfg(feature = "nightly")]
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, &'a T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<'a, T, U, S: Shape, L: Layout, I: Apply<U>> $trt<I> for &'a Slice<T, S, L>
        where
            &'a T: $trt<I::Item, Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = I::ZippedWith<Self, fn((I::Item, &'a T)) -> U>;

            #[cfg(feature = "nightly")]
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, &'a T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<'a, T, U, S: Shape, A: Allocator, I: Apply<U>> $trt<I> for &'a Tensor<T, S, A>
        where
            &'a T: $trt<I::Item, Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = I::ZippedWith<Self, fn((I::Item, &'a T)) -> U>;

            #[cfg(feature = "nightly")]
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, &'a T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<'a, T, U, S: Shape, L: Layout, I: Apply<U>> $trt<I> for &'a View<'_, T, S, L>
        where
            &'a T: $trt<I::Item, Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = I::ZippedWith<Self, fn((I::Item, &'a T)) -> U>;

            #[cfg(feature = "nightly")]
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, &'a T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<'a, T, U, S: Shape, L: Layout, I: Apply<U>> $trt<I> for &'a ViewMut<'_, T, S, L>
        where
            &'a T: $trt<I::Item, Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = I::ZippedWith<Self, fn((I::Item, &'a T)) -> U>;

            #[cfg(feature = "nightly")]
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, &'a T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<T, U, S: ConstShape, I: IntoExpression> $trt<I> for Array<T, S>
        where
            T: $trt<I::Item, Output = U>,
        {
            type Output = Array<U, S>;

            fn $fn(self, rhs: I) -> Self::Output {
                self.zip_with(rhs, |(x, y)| x.$fn(y))
            }
        }

        impl<T: Clone, U, I: Apply<U>> $trt<I> for Fill<T>
        where
            T: $trt<I::Item, Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = I::ZippedWith<Self, fn((I::Item, T)) -> U>;

            #[cfg(feature = "nightly")]
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<T: Clone, U, F: FnMut() -> T, I: Apply<U>> $trt<I> for FillWith<F>
        where
            T: $trt<I::Item, Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = I::ZippedWith<Self, fn((I::Item, T)) -> U>;

            #[cfg(feature = "nightly")]
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<S: Shape, T: Clone, U, I: Apply<U>> $trt<I> for FromElem<T, S>
        where
            T: $trt<I::Item, Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = I::ZippedWith<Self, fn((I::Item, T)) -> U>;

            #[cfg(feature = "nightly")]
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<S: Shape, T, U, F: FnMut(S::Dims) -> T, I: Apply<U>> $trt<I> for FromFn<S, F>
        where
            T: $trt<I::Item, Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = I::ZippedWith<Self, fn((I::Item, T)) -> U>;

            #[cfg(feature = "nightly")]
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<T, B: Buffer, I: Apply<T>> $trt<I> for IntoExpr<B>
        where
            B::Item: $trt<I::Item, Output = T>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = I::ZippedWith<Self, fn((I::Item, B::Item)) -> T>;

            #[cfg(feature = "nightly")]
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, B::Item)) -> T>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<T, U, E: Expression, F: FnMut(E::Item) -> T, I: Apply<U>> $trt<I> for Map<E, F>
        where
            T: $trt<I::Item, Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = I::ZippedWith<Self, fn((I::Item, T)) -> U>;

            #[cfg(feature = "nightly")]
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<T: Default, S: Shape, A: Allocator, I: IntoExpression> $trt<I> for Tensor<T, S, A>
        where
            T: $trt<I::Item, Output = T>,
        {
            type Output = Self;

            fn $fn(self, rhs: I) -> Self {
                self.zip_with(rhs, |(x, y)| x.$fn(y))
            }
        }

        impl<'a, T, U, S: Shape, L: Layout, I: Apply<U>> $trt<I> for View<'a, T, S, L>
        where
            &'a T: $trt<I::Item, Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = I::ZippedWith<Self, fn((I::Item, &'a T)) -> U>;

            #[cfg(feature = "nightly")]
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, &'a T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
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
        impl<T, S: ConstShape, I: IntoExpression> $trt<I> for Array<T, S>
        where
            T: $trt<I::Item>,
        {
            fn $fn(&mut self, rhs: I) {
                self.expr_mut().zip(rhs).for_each(|(x, y)| x.$fn(y));
            }
        }

        impl<T, S: Shape, L: Layout, I: IntoExpression> $trt<I> for Slice<T, S, L>
        where
            T: $trt<I::Item>,
        {
            fn $fn(&mut self, rhs: I) {
                self.expr_mut().zip(rhs).for_each(|(x, y)| x.$fn(y));
            }
        }

        impl<T, S: Shape, A: Allocator, I: IntoExpression> $trt<I> for Tensor<T, S, A>
        where
            T: $trt<I::Item>,
        {
            fn $fn(&mut self, rhs: I) {
                self.expr_mut().zip(rhs).for_each(|(x, y)| x.$fn(y));
            }
        }

        impl<T, S: Shape, L: Layout, I: IntoExpression> $trt<I> for ViewMut<'_, T, S, L>
        where
            T: $trt<I::Item>,
        {
            fn $fn(&mut self, rhs: I) {
                self.expr_mut().zip(rhs).for_each(|(x, y)| x.$fn(y));
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
        impl<'a, T, U, S: ConstShape> $trt for &'a Array<T, S>
        where
            &'a T: $trt<Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = <Self as Apply<U>>::Output<fn(&'a T) -> U>;

            #[cfg(feature = "nightly")]
            type Output = <Self as Apply<U>>::Output<impl FnMut(&'a T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<'a, T, U, S: Shape, L: Layout> $trt for &'a Slice<T, S, L>
        where
            &'a T: $trt<Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = <Self as Apply<U>>::Output<fn(&'a T) -> U>;

            #[cfg(feature = "nightly")]
            type Output = <Self as Apply<U>>::Output<impl FnMut(&'a T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<'a, T, U, S: Shape, A: Allocator> $trt for &'a Tensor<T, S, A>
        where
            &'a T: $trt<Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = <Self as Apply<U>>::Output<fn(&'a T) -> U>;

            #[cfg(feature = "nightly")]
            type Output = <Self as Apply<U>>::Output<impl FnMut(&'a T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<'a, T, U, S: Shape, L: Layout> $trt for &'a View<'_, T, S, L>
        where
            &'a T: $trt<Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = <Self as Apply<U>>::Output<fn(&'a T) -> U>;

            #[cfg(feature = "nightly")]
            type Output = <Self as Apply<U>>::Output<impl FnMut(&'a T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<'a, T, U, S: Shape, L: Layout> $trt for &'a ViewMut<'_, T, S, L>
        where
            &'a T: $trt<Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = <Self as Apply<U>>::Output<fn(&'a T) -> U>;

            #[cfg(feature = "nightly")]
            type Output = <Self as Apply<U>>::Output<impl FnMut(&'a T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<T, U, S: ConstShape> $trt for Array<T, S>
        where
            T: $trt<Output = U>,
        {
            type Output = Array<U, S>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<T: Clone, U> $trt for Fill<T>
        where
            T: $trt<Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = <Self as Apply<U>>::Output<fn(T) -> U>;

            #[cfg(feature = "nightly")]
            type Output = <Self as Apply<U>>::Output<impl FnMut(T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<T: Clone, U, F: FnMut() -> T> $trt for FillWith<F>
        where
            T: $trt<Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = <Self as Apply<U>>::Output<fn(T) -> U>;

            #[cfg(feature = "nightly")]
            type Output = <Self as Apply<U>>::Output<impl FnMut(T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<S: Shape, T: Clone, U> $trt for FromElem<T, S>
        where
            T: $trt<Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = <Self as Apply<U>>::Output<fn(T) -> U>;

            #[cfg(feature = "nightly")]
            type Output = <Self as Apply<U>>::Output<impl FnMut(T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<S: Shape, T, U, F: FnMut(S::Dims) -> T> $trt for FromFn<S, F>
        where
            T: $trt<Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = <Self as Apply<U>>::Output<fn(T) -> U>;

            #[cfg(feature = "nightly")]
            type Output = <Self as Apply<U>>::Output<impl FnMut(T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<T, B: Buffer> $trt for IntoExpr<B>
        where
            B::Item: $trt<Output = T>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = <Self as Apply<T>>::Output<fn(B::Item) -> T>;

            #[cfg(feature = "nightly")]
            type Output = <Self as Apply<T>>::Output<impl FnMut(B::Item) -> T>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<T, U, E: Expression, F: FnMut(E::Item) -> T> $trt for Map<E, F>
        where
            T: $trt<Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = <Self as Apply<U>>::Output<fn(T) -> U>;

            #[cfg(feature = "nightly")]
            type Output = <Self as Apply<U>>::Output<impl FnMut(T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<T: Default, S: Shape, A: Allocator> $trt for Tensor<T, S, A>
        where
            T: $trt<Output = T>,
        {
            type Output = Self;

            fn $fn(self) -> Self {
                self.apply(|x| x.$fn())
            }
        }

        impl<'a, T, U, S: Shape, L: Layout> $trt for View<'a, T, S, L>
        where
            &'a T: $trt<Output = U>,
        {
            #[cfg(not(feature = "nightly"))]
            type Output = <Self as Apply<U>>::Output<fn(&'a T) -> U>;

            #[cfg(feature = "nightly")]
            type Output = <Self as Apply<U>>::Output<impl FnMut(&'a T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }
    };
}

impl_unary_op!(Neg, neg);
impl_unary_op!(Not, not);
