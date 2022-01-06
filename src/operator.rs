use std::cmp::Ordering;

use crate::buffer::Buffer;
use crate::dimension::{Const, Dim};
use crate::grid::GridBase;
use crate::layout::Layout;
use crate::order::Order;
use crate::span::SpanBase;

impl<B: Buffer> Eq for GridBase<B> where SpanBase<B::Item, B::Layout>: Eq {}
impl<T: Eq, L: Layout> Eq for SpanBase<T, L> {}

impl<B: Buffer> Ord for GridBase<B>
where
    SpanBase<B::Item, B::Layout>: Ord,
{
    fn cmp(&self, other: &GridBase<B>) -> Ordering {
        self.as_span().cmp(other.as_span())
    }
}

impl<T: Ord, L: Layout<Dim = Const<1>>> Ord for SpanBase<T, L> {
    fn cmp(&self, other: &SpanBase<T, L>) -> Ordering {
        if L::IS_UNIT_STRIDED {
            self.as_slice().cmp(other.as_slice())
        } else {
            self.iter().cmp(other)
        }
    }
}

impl<B: Buffer, C: Buffer> PartialOrd<GridBase<C>> for GridBase<B>
where
    SpanBase<B::Item, B::Layout>: PartialOrd<SpanBase<C::Item, C::Layout>>,
{
    fn partial_cmp(&self, other: &GridBase<C>) -> Option<Ordering> {
        self.as_span().partial_cmp(other.as_span())
    }
}

impl<B: Buffer, T, L: Layout> PartialOrd<SpanBase<T, L>> for GridBase<B>
where
    SpanBase<B::Item, B::Layout>: PartialOrd<SpanBase<T, L>>,
{
    fn partial_cmp(&self, other: &SpanBase<T, L>) -> Option<Ordering> {
        self.as_span().partial_cmp(other)
    }
}

impl<T, L: Layout, B: Buffer> PartialOrd<GridBase<B>> for SpanBase<T, L>
where
    SpanBase<T, L>: PartialOrd<SpanBase<B::Item, B::Layout>>,
{
    fn partial_cmp(&self, other: &GridBase<B>) -> Option<Ordering> {
        self.partial_cmp(other.as_span())
    }
}

impl<O: Order, T: PartialOrd<U>, L, U, M> PartialOrd<SpanBase<U, M>> for SpanBase<T, L>
where
    L: Layout<Dim = Const<1>, Order = O>,
    M: Layout<Dim = Const<1>, Order = O>,
{
    fn partial_cmp(&self, other: &SpanBase<U, M>) -> Option<Ordering> {
        self.iter().partial_cmp(other)
    }
}

impl<B: Buffer, C: Buffer> PartialEq<GridBase<C>> for GridBase<B>
where
    SpanBase<B::Item, B::Layout>: PartialEq<SpanBase<C::Item, C::Layout>>,
{
    fn eq(&self, other: &GridBase<C>) -> bool {
        self.as_span().eq(other.as_span())
    }
}

impl<B: Buffer, T, L: Layout> PartialEq<SpanBase<T, L>> for GridBase<B>
where
    SpanBase<B::Item, B::Layout>: PartialEq<SpanBase<T, L>>,
{
    fn eq(&self, other: &SpanBase<T, L>) -> bool {
        self.as_span().eq(other)
    }
}

impl<T, L: Layout, B: Buffer> PartialEq<GridBase<B>> for SpanBase<T, L>
where
    SpanBase<T, L>: PartialEq<SpanBase<B::Item, B::Layout>>,
{
    fn eq(&self, other: &GridBase<B>) -> bool {
        self.eq(other.as_span())
    }
}

impl<D: Dim, O: Order, T: PartialEq<U>, L, U, M> PartialEq<SpanBase<U, M>> for SpanBase<T, L>
where
    L: Layout<Dim = D, Order = O>,
    M: Layout<Dim = D, Order = O>,
{
    fn eq(&self, other: &SpanBase<U, M>) -> bool {
        if D::RANK == 0 {
            self[D::Shape::default()] == other[D::Shape::default()]
        } else if D::RANK == 1 {
            if L::IS_UNIT_STRIDED && M::IS_UNIT_STRIDED {
                self.as_slice().eq(other.as_slice())
            } else {
                self.flat_iter().eq(other.flat_iter())
            }
        } else if L::IS_DENSE && M::IS_DENSE {
            self.shape().as_ref() == other.shape().as_ref() && self.as_slice().eq(other.as_slice())
        } else {
            self.outer_iter().eq(other.outer_iter())
        }
    }
}
