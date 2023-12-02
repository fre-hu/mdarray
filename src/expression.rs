#[cfg(feature = "nightly")]
use std::alloc::Allocator;
use std::fmt::{Debug, Formatter, Result};

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::array::GridArray;
use crate::dim::Dim;
use crate::expr::{Cloned, Copied, Enumerate, Map, Producer, Zip};
use crate::iter::Iter;
use crate::traits::{Apply, IntoCloned, IntoExpression};

/// Expression type, for multidimensional iteration.
#[derive(Clone)]
pub struct Expression<P> {
    producer: P,
}

impl<P: Producer> Expression<P> {
    /// Creates an expression which clones all of its elements.
    pub fn cloned<'a, T: 'a + Clone>(self) -> Expression<Cloned<P>>
    where
        P: Producer<Item = &'a T>,
    {
        Expression::new(Cloned::new(self.into_producer()))
    }

    /// Creates an expression which copies all of its elements.
    pub fn copied<'a, T: 'a + Copy>(self) -> Expression<Copied<P>>
    where
        P: Producer<Item = &'a T>,
    {
        Expression::new(Copied::new(self.into_producer()))
    }

    /// Creates an expression which gives tuples of the current index and the element.
    pub fn enumerate(self) -> Expression<Enumerate<P, <P::Dim as Dim>::Shape>> {
        Expression::new(Enumerate::new(self.into_producer()))
    }

    /// Evaluates the expression into a new array.
    pub fn eval(self) -> GridArray<P::Item, P::Dim> {
        GridArray::from_expr(self)
    }

    /// Evaluates the expression into a new array with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn eval_in<A: Allocator>(self, alloc: A) -> GridArray<P::Item, P::Dim, A> {
        GridArray::from_expr_in(self, alloc)
    }

    /// Evaluates the expression with broadcasting and appends to the given array
    /// along the outermost dimension.
    ///
    /// If the rank of the expression equals one less than the rank of the array,
    /// the expression is assumed to have outermost dimension of size 1.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match, or if the rank of the expression
    /// is not valid.
    pub fn eval_into<D: Dim, A: Allocator>(
        self,
        grid: &mut GridArray<P::Item, D, A>,
    ) -> &mut GridArray<P::Item, D, A> {
        grid.expand(self);
        grid
    }

    /// Folds all elements into an accumulator by applying an operation, and returns the result.
    pub fn fold<T, F: FnMut(T, P::Item) -> T>(self, init: T, f: F) -> T {
        Iter::new(self.into_producer()).fold(init, f)
    }

    /// Calls a closure on each element of the expression.
    pub fn for_each<F: FnMut(P::Item)>(self, mut f: F) {
        self.fold((), |(), x| f(x));
    }

    /// Returns `true` if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.shape()[..].iter().product()
    }

    /// Creates an expression that calls a closure on each element.
    pub fn map<T, F: FnMut(P::Item) -> T>(self, f: F) -> Expression<Map<P, F>> {
        Expression::new(Map::new(self.into_producer(), f))
    }

    /// Returns the array rank, i.e. the number of dimensions.
    pub fn rank(&self) -> usize {
        P::Dim::RANK
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> <P::Dim as Dim>::Shape {
        self.producer.shape()
    }

    /// Returns the number of elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn size(&self, dim: usize) -> usize {
        assert!(dim < P::Dim::RANK, "invalid dimension");

        self.shape()[dim]
    }

    /// Creates an expression that gives tuples `(x, y)` of the elements from each expression.
    ///
    /// # Panics
    ///
    /// Panics if the expressions cannot be broadcast to a common shape.
    pub fn zip<I: IntoExpression>(self, other: I) -> Expression<Zip<P, I::Producer>> {
        Expression::new(Zip::new(self.into_producer(), other.into_expr().into_producer()))
    }

    #[cfg(not(feature = "nightly"))]
    pub(crate) fn clone_into_vec<T>(self, vec: &mut Vec<T>)
    where
        P::Item: IntoCloned<T>,
    {
        assert!(self.len() <= vec.capacity() - vec.len(), "length exceeds capacity");

        self.for_each(|x| unsafe {
            vec.as_mut_ptr().add(vec.len()).write(x.into_cloned());
            vec.set_len(vec.len() + 1);
        });
    }

    #[cfg(feature = "nightly")]
    pub(crate) fn clone_into_vec<T, A: Allocator>(self, vec: &mut Vec<T, A>)
    where
        P::Item: IntoCloned<T>,
    {
        assert!(self.len() <= vec.capacity() - vec.len(), "length exceeds capacity");

        self.for_each(|x| unsafe {
            vec.as_mut_ptr().add(vec.len()).write(x.into_cloned());
            vec.set_len(vec.len() + 1);
        });
    }

    pub(crate) fn into_producer(self) -> P {
        self.producer
    }

    pub(crate) fn new(producer: P) -> Self {
        Self { producer }
    }
}

impl<T, P: Producer> Apply<T> for Expression<P> {
    type Output<F: FnMut(Self::Item) -> T> = Expression<impl Producer<Item = T, Dim = P::Dim>>;
    type ZippedWith<I: IntoExpression, F: FnMut(Self::Item, I::Item) -> T> =
        Expression<impl Producer<Item = T, Dim = <P::Dim as Dim>::Max<I::Dim>>>;

    fn apply<F: FnMut(Self::Item) -> T>(self, f: F) -> Self::Output<F> {
        self.map(f)
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, mut f: F) -> Self::ZippedWith<I, F>
    where
        F: FnMut(Self::Item, I::Item) -> T,
    {
        self.zip(expr).map(move |(x, y)| f(x, y))
    }
}

impl<P: Producer + Debug> Debug for Expression<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        self.producer.fmt(f)
    }
}

impl<P: Producer> IntoExpression for Expression<P> {
    type Item = P::Item;
    type Dim = P::Dim;
    type Producer = P;

    fn into_expr(self) -> Self {
        self
    }
}

impl<P: Producer> IntoIterator for Expression<P> {
    type Item = P::Item;
    type IntoIter = Iter<P>;

    fn into_iter(self) -> Iter<P> {
        Iter::new(self.producer)
    }
}
