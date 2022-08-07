use std::alloc::Allocator;
use std::cmp::Ordering;
use std::mem;

use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

use crate::buffer::{Buffer, BufferMut};
use crate::dim::{Dim, U1};
use crate::format::Format;
use crate::grid::{DenseGrid, GridBase};
use crate::layout::{DenseLayout, Layout};
use crate::order::Order;
use crate::span::SpanBase;

/// Fill value to be used as scalar operand for array operators.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fill<T: Copy> {
    value: T,
}

/// Returns a fill value to be used as scalar operand for array operators.
///
/// If the type does not implement the `Copy` trait, a reference must be passed as input.
pub fn fill<T: Copy>(value: T) -> Fill<T> {
    Fill { value }
}

impl<T: Eq, B: Buffer<Item = T>> Eq for GridBase<B> where GridBase<B>: PartialEq {}
impl<T: Eq, L: Copy> Eq for SpanBase<T, L> where SpanBase<T, L>: PartialEq {}

impl<T: Ord, B: Buffer<Item = T, Layout = Layout<U1, impl Format, impl Order>>> Ord
    for GridBase<B>
{
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.as_span().cmp(rhs.as_span())
    }
}

impl<T: Ord, F: Format, O: Order> Ord for SpanBase<T, Layout<U1, F, O>> {
    fn cmp(&self, rhs: &Self) -> Ordering {
        if self.has_slice_indexing() {
            self.as_slice().cmp(rhs.as_slice())
        } else {
            self.flatten().iter().cmp(rhs.flatten().iter())
        }
    }
}

impl<B: Buffer, X: ?Sized> PartialOrd<X> for GridBase<B>
where
    SpanBase<B::Item, B::Layout>: PartialOrd<X>,
{
    fn partial_cmp(&self, other: &X) -> Option<Ordering> {
        self.as_span().partial_cmp(other)
    }
}

impl<T, F: Format, O: Order, B: Buffer<Layout = Layout<U1, impl Format, O>>> PartialOrd<GridBase<B>>
    for SpanBase<T, Layout<U1, F, O>>
where
    T: PartialOrd<B::Item>,
{
    fn partial_cmp(&self, rhs: &GridBase<B>) -> Option<Ordering> {
        self.partial_cmp(rhs.as_span())
    }
}

impl<T, U, F: Format, G: Format, O: Order> PartialOrd<SpanBase<U, Layout<U1, G, O>>>
    for SpanBase<T, Layout<U1, F, O>>
where
    T: PartialOrd<U>,
{
    fn partial_cmp(&self, rhs: &SpanBase<U, Layout<U1, G, O>>) -> Option<Ordering> {
        self.flatten().iter().partial_cmp(rhs.flatten().iter())
    }
}

impl<B: Buffer, X: ?Sized> PartialEq<X> for GridBase<B>
where
    SpanBase<B::Item, B::Layout>: PartialEq<X>,
{
    fn eq(&self, other: &X) -> bool {
        self.as_span().eq(other)
    }
}

impl<T, D: Dim, F: Format, O: Order, B: Buffer<Layout = Layout<D, impl Format, O>>>
    PartialEq<GridBase<B>> for SpanBase<T, Layout<D, F, O>>
where
    T: PartialEq<B::Item>,
{
    fn eq(&self, rhs: &GridBase<B>) -> bool {
        self.eq(rhs.as_span())
    }
}

impl<T, U, D: Dim, F: Format, G: Format, O: Order> PartialEq<SpanBase<U, Layout<D, G, O>>>
    for SpanBase<T, Layout<D, F, O>>
where
    T: PartialEq<U>,
{
    fn eq(&self, rhs: &SpanBase<U, Layout<D, G, O>>) -> bool {
        if self.has_slice_indexing() && rhs.has_slice_indexing() {
            self.shape()[..] == rhs.shape()[..] && self.as_slice().eq(rhs.as_slice())
        } else if self.has_linear_indexing() && rhs.has_linear_indexing() {
            self.shape()[..] == rhs.shape()[..] && self.flatten().iter().eq(rhs.flatten().iter())
        } else {
            self.outer_iter().eq(rhs.outer_iter())
        }
    }
}

macro_rules! impl_binary_op {
    ($trt:tt, $fn:tt) => {
        impl<'a, B: Buffer, X> $trt<X> for &'a GridBase<B>
        where
            &'a SpanBase<B::Item, B::Layout>: $trt<X>,
        {
            type Output = <&'a SpanBase<B::Item, B::Layout> as $trt<X>>::Output;

            fn $fn(self, rhs: X) -> Self::Output {
                self.as_span().$fn(rhs)
            }
        }

        impl<T, U, D: Dim, F: Format, O: Order, B: Buffer<Layout = Layout<D, impl Format, O>>>
            $trt<&GridBase<B>> for &SpanBase<T, Layout<D, F, O>>
        where
            for<'a, 'b> &'a T: $trt<&'b B::Item, Output = U>,
        {
            type Output = DenseGrid<U, D, O>;

            fn $fn(self, rhs: &GridBase<B>) -> Self::Output {
                self.$fn(rhs.as_span())
            }
        }

        impl<T, U, V, D: Dim, F: Format, G: Format, O: Order> $trt<&SpanBase<U, Layout<D, G, O>>>
            for &SpanBase<T, Layout<D, F, O>>
        where
            for<'a, 'b> &'a T: $trt<&'b U, Output = V>,
        {
            type Output = DenseGrid<V, D, O>;

            fn $fn(self, rhs: &SpanBase<U, Layout<D, G, O>>) -> Self::Output {
                let mut vec = Vec::with_capacity(self.len());

                unsafe {
                    from_binary_op(self, rhs, &mut vec, &|x, y| x.$fn(y));

                    DenseGrid::from_parts(vec, DenseLayout::new(self.shape()))
                }
            }
        }

        impl<T: Copy, U, D: Dim, O: Order, B: Buffer<Layout = Layout<D, impl Format, O>>>
            $trt<&GridBase<B>> for Fill<T>
        where
            for<'a> T: $trt<&'a B::Item, Output = U>,
        {
            type Output = DenseGrid<U, D, O>;

            fn $fn(self, rhs: &GridBase<B>) -> Self::Output {
                self.$fn(rhs.as_span())
            }
        }

        impl<T: Copy, U, V, D: Dim, F: Format, O: Order> $trt<&SpanBase<U, Layout<D, F, O>>>
            for Fill<T>
        where
            for<'a> T: $trt<&'a U, Output = V>,
        {
            type Output = DenseGrid<V, D, O>;

            fn $fn(self, rhs: &SpanBase<U, Layout<D, F, O>>) -> Self::Output {
                let mut vec = Vec::with_capacity(rhs.len());

                unsafe {
                    from_unary_op(rhs, &mut vec, &|x| self.value.$fn(x));

                    DenseGrid::from_parts(vec, DenseLayout::new(rhs.shape()))
                }
            }
        }

        impl<T, U: Copy, V, D: Dim, F: Format, O: Order> $trt<Fill<U>>
            for &SpanBase<T, Layout<D, F, O>>
        where
            for<'a> &'a T: $trt<U, Output = V>,
        {
            type Output = DenseGrid<V, D, O>;

            fn $fn(self, rhs: Fill<U>) -> Self::Output {
                let mut vec = Vec::with_capacity(self.len());

                unsafe {
                    from_unary_op(self, &mut vec, &|x| x.$fn(rhs.value));

                    DenseGrid::from_parts(vec, DenseLayout::new(self.shape()))
                }
            }
        }

        impl<T: Default, D: Dim, O: Order, B: Buffer<Layout = Layout<D, impl Format, O>>, A>
            $trt<&GridBase<B>> for DenseGrid<T, D, O, A>
        where
            for<'a> T: $trt<&'a B::Item, Output = T>,
            A: Allocator,
        {
            type Output = Self;

            fn $fn(self, rhs: &GridBase<B>) -> Self {
                self.$fn(rhs.as_span())
            }
        }

        impl<T: Default, U, D: Dim, F: Format, O: Order, A> $trt<&SpanBase<U, Layout<D, F, O>>>
            for DenseGrid<T, D, O, A>
        where
            for<'a> T: $trt<&'a U, Output = T>,
            A: Allocator,
        {
            type Output = Self;

            fn $fn(mut self, rhs: &SpanBase<U, Layout<D, F, O>>) -> Self {
                map_binary_op(&mut self, rhs, &|(x, y)| *x = mem::take(x).$fn(y));

                self
            }
        }

        impl<T: Default, U: Copy, D: Dim, O: Order, A: Allocator> $trt<Fill<U>>
            for DenseGrid<T, D, O, A>
        where
            T: $trt<U, Output = T>,
        {
            type Output = Self;

            fn $fn(mut self, rhs: Fill<U>) -> Self {
                map_unary_op(&mut self, &|x| *x = mem::take(x).$fn(rhs.value));

                self
            }
        }

        impl<T, U: Default, D: Dim, F: Format, O: Order, A: Allocator> $trt<DenseGrid<U, D, O, A>>
            for &SpanBase<T, Layout<D, F, O>>
        where
            for<'a> &'a T: $trt<U, Output = U>,
        {
            type Output = DenseGrid<U, D, O, A>;

            fn $fn(self, mut rhs: DenseGrid<U, D, O, A>) -> Self::Output {
                map_binary_op(&mut rhs, self, &|(x, y)| *x = y.$fn(mem::take(x)));

                rhs
            }
        }

        impl<T: Copy, U: Default, D: Dim, O: Order, A: Allocator> $trt<DenseGrid<U, D, O, A>>
            for Fill<T>
        where
            T: $trt<U, Output = U>,
        {
            type Output = DenseGrid<U, D, O, A>;

            fn $fn(self, mut rhs: DenseGrid<U, D, O, A>) -> Self::Output {
                map_unary_op(&mut rhs, &|x| *x = self.value.$fn(mem::take(x)));

                rhs
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
        impl<B: BufferMut, X> $trt<X> for GridBase<B>
        where
            SpanBase<B::Item, B::Layout>: $trt<X>,
        {
            fn $fn(&mut self, rhs: X) {
                self.as_mut_span().$fn(rhs);
            }
        }

        impl<T, D: Dim, F: Format, O: Order, B: Buffer<Layout = Layout<D, impl Format, O>>>
            $trt<&GridBase<B>> for SpanBase<T, Layout<D, F, O>>
        where
            for<'a> T: $trt<&'a B::Item>,
        {
            fn $fn(&mut self, rhs: &GridBase<B>) {
                self.$fn(rhs.as_span());
            }
        }

        impl<T, U, D: Dim, F: Format, G: Format, O: Order> $trt<&SpanBase<U, Layout<D, G, O>>>
            for SpanBase<T, Layout<D, F, O>>
        where
            for<'a> T: $trt<&'a U>,
        {
            fn $fn(&mut self, rhs: &SpanBase<U, Layout<D, G, O>>) {
                map_binary_op(self, rhs, &|(x, y)| x.$fn(y));
            }
        }

        impl<T, U: Copy, D: Dim, F: Format, O: Order> $trt<Fill<U>> for SpanBase<T, Layout<D, F, O>>
        where
            T: $trt<U>,
        {
            fn $fn(&mut self, rhs: Fill<U>) {
                map_unary_op(self, &|x| x.$fn(rhs.value));
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
        impl<D: Dim, O: Order, B: Buffer<Layout = Layout<D, impl Format, O>>> $trt for &GridBase<B>
        where
            for<'a> &'a B::Item: $trt<Output = B::Item>,
        {
            type Output = DenseGrid<B::Item, D, O>;

            fn $fn(self) -> Self::Output {
                self.as_span().$fn()
            }
        }

        impl<T, D: Dim, F: Format, O: Order> $trt for &SpanBase<T, Layout<D, F, O>>
        where
            for<'a> &'a T: $trt<Output = T>,
        {
            type Output = DenseGrid<T, D, O>;

            fn $fn(self) -> Self::Output {
                let mut vec = Vec::with_capacity(self.len());

                unsafe {
                    from_unary_op(self, &mut vec, &|x| x.$fn());

                    DenseGrid::from_parts(vec, DenseLayout::new(self.shape()))
                }
            }
        }

        impl<T: Default, D: Dim, O: Order, A: Allocator> $trt for DenseGrid<T, D, O, A>
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

unsafe fn from_binary_op<T, U, V, D: Dim, O: Order, F: Fn(&T, &U) -> V>(
    lhs: &SpanBase<T, Layout<D, impl Format, O>>,
    rhs: &SpanBase<U, Layout<D, impl Format, O>>,
    vec: &mut Vec<V>,
    f: &F,
) {
    if lhs.has_linear_indexing() && rhs.has_linear_indexing() {
        assert!(lhs.shape()[..] == rhs.shape()[..], "shape mismatch");

        for (x, y) in lhs.flatten().iter().zip(rhs.flatten().iter()) {
            vec.as_mut_ptr().add(vec.len()).write(f(x, y));
            vec.set_len(vec.len() + 1);
        }
    } else {
        let dim = D::dim::<O>(D::RANK - 1);

        assert!(lhs.size(dim) == rhs.size(dim), "shape mismatch");

        for (x, y) in lhs.outer_iter().zip(rhs.outer_iter()) {
            from_binary_op(&x, &y, vec, f);
        }
    }
}

unsafe fn from_unary_op<T, U, F: Fn(&T) -> U>(
    span: &SpanBase<T, Layout<impl Dim, impl Format, impl Order>>,
    vec: &mut Vec<U>,
    f: &F,
) {
    if span.has_linear_indexing() {
        for x in span.flatten().iter() {
            vec.as_mut_ptr().add(vec.len()).write(f(x));
            vec.set_len(vec.len() + 1);
        }
    } else {
        for x in span.outer_iter() {
            from_unary_op(&x, vec, f);
        }
    }
}

fn map_binary_op<T, U, D: Dim, O: Order, F: Fn((&mut T, &U))>(
    lhs: &mut SpanBase<T, Layout<D, impl Format, O>>,
    rhs: &SpanBase<U, Layout<D, impl Format, O>>,
    f: &F,
) {
    if lhs.has_linear_indexing() && rhs.has_linear_indexing() {
        assert!(lhs.shape()[..] == rhs.shape()[..], "shape mismatch");

        lhs.flatten_mut().iter_mut().zip(rhs.flatten().iter()).for_each(f);
    } else {
        let dim = D::dim::<O>(D::RANK - 1);

        assert!(lhs.size(dim) == rhs.size(dim), "shape mismatch");

        for (mut x, y) in lhs.outer_iter_mut().zip(rhs.outer_iter()) {
            map_binary_op(&mut x, &y, f);
        }
    }
}

fn map_unary_op<T>(
    span: &mut SpanBase<T, Layout<impl Dim, impl Format, impl Order>>,
    f: &impl Fn(&mut T),
) {
    if span.has_linear_indexing() {
        span.flatten_mut().iter_mut().for_each(f);
    } else {
        for mut x in span.outer_iter_mut() {
            map_unary_op(&mut x, f);
        }
    }
}
