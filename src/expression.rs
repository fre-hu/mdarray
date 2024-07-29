#[cfg(feature = "nightly")]
use std::alloc::Allocator;

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::expr::{Cloned, Copied, Enumerate, Map, Zip};
use crate::grid::Grid;
use crate::iter::Iter;
use crate::shape::Shape;
use crate::traits::{Apply, IntoCloned, IntoExpression};

/// Expression trait, for multidimensional iteration.
pub trait Expression: IntoIterator {
    /// Array shape type.
    type Shape: Shape;

    /// True if the expression can be restarted from the beginning after the last element.
    const IS_REPEATABLE: bool;

    /// Bitmask per dimension, indicating if it must not be merged with its outer dimension.
    const SPLIT_MASK: usize;

    /// Returns the array shape.
    fn shape(&self) -> Self::Shape;

    /// Creates an expression which clones all of its elements.
    fn cloned<'a, T: 'a + Clone>(self) -> Cloned<Self>
    where
        Self: Expression<Item = &'a T> + Sized,
    {
        Cloned::new(self)
    }

    /// Creates an expression which copies all of its elements.
    fn copied<'a, T: 'a + Copy>(self) -> Copied<Self>
    where
        Self: Expression<Item = &'a T> + Sized,
    {
        Copied::new(self)
    }

    /// Returns the number of elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    fn dim(&self, index: usize) -> usize {
        self.shape().dim(index)
    }

    /// Returns the number of elements in each dimension.
    fn dims(&self) -> <Self::Shape as Shape>::Dims {
        self.shape().dims()
    }

    /// Creates an expression which gives tuples of the current index and the element.
    fn enumerate(self) -> Enumerate<Self, <Self::Shape as Shape>::Dims>
    where
        Self: Sized,
    {
        Enumerate::new(self)
    }

    /// Evaluates the expression into a new array.
    fn eval(self) -> Grid<Self::Item, Self::Shape>
    where
        Self: Sized,
    {
        Grid::from_expr(self)
    }

    /// Evaluates the expression into a new array with the specified allocator.
    #[cfg(feature = "nightly")]
    fn eval_in<A: Allocator>(self, alloc: A) -> Grid<Self::Item, Self::Shape, A>
    where
        Self: Sized,
    {
        Grid::from_expr_in(self, alloc)
    }

    /// Evaluates the expression with broadcasting and appends to the given array
    /// along the outermost dimension.
    ///
    /// If the rank of the expression equals one less than the rank of the array,
    /// the expression is assumed to have outermost dimension of size 1.
    ///
    /// If the array is empty, it is reshaped to match the shape of the expression.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match, if the rank of the expression
    /// is not valid, or if the outermost dimension is not dynamically-sized.
    fn eval_into<S: Shape, A: Allocator>(
        self,
        grid: &mut Grid<Self::Item, S, A>,
    ) -> &mut Grid<Self::Item, S, A>
    where
        Self: Sized,
    {
        grid.expand(self);
        grid
    }

    /// Folds all elements into an accumulator by applying an operation, and returns the result.
    fn fold<T, F: FnMut(T, Self::Item) -> T>(self, init: T, f: F) -> T
    where
        Self: Sized,
    {
        Iter::new(self).fold(init, f)
    }

    /// Calls a closure on each element of the expression.
    fn for_each<F: FnMut(Self::Item)>(self, mut f: F)
    where
        Self: Sized,
    {
        self.fold((), |(), x| f(x));
    }

    /// Returns `true` if the array contains no elements.
    fn is_empty(&self) -> bool {
        self.shape().is_empty()
    }

    /// Returns the number of elements in the array.
    fn len(&self) -> usize {
        self.shape().len()
    }

    /// Creates an expression that calls a closure on each element.
    fn map<T, F: FnMut(Self::Item) -> T>(self, f: F) -> Map<Self, F>
    where
        Self: Sized,
    {
        Map::new(self, f)
    }

    /// Returns the array rank, i.e. the number of dimensions.
    fn rank(&self) -> usize {
        Self::Shape::RANK
    }

    /// Creates an expression that gives tuples `(x, y)` of the elements from each expression.
    ///
    /// # Panics
    ///
    /// Panics if the expressions cannot be broadcast to a common shape.
    fn zip<I: IntoExpression>(self, other: I) -> Zip<Self, I::IntoExpr>
    where
        Self: Sized,
    {
        Zip::new(self, other.into_expr())
    }

    #[doc(hidden)]
    unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item;

    #[doc(hidden)]
    unsafe fn reset_dim(&mut self, index: usize, count: usize);

    #[doc(hidden)]
    unsafe fn step_dim(&mut self, index: usize);

    #[cfg(not(feature = "nightly"))]
    #[doc(hidden)]
    fn clone_into_vec<T>(self, vec: &mut Vec<T>)
    where
        Self: Expression<Item: IntoCloned<T>> + Sized,
    {
        assert!(self.len() <= vec.capacity() - vec.len(), "length exceeds capacity");

        self.for_each(|x| unsafe {
            vec.as_mut_ptr().add(vec.len()).write(x.into_cloned());
            vec.set_len(vec.len() + 1);
        });
    }

    #[cfg(feature = "nightly")]
    #[doc(hidden)]
    fn clone_into_vec<T, A: Allocator>(self, vec: &mut Vec<T, A>)
    where
        Self: Expression<Item: IntoCloned<T>> + Sized,
    {
        assert!(self.len() <= vec.capacity() - vec.len(), "length exceeds capacity");

        self.for_each(|x| unsafe {
            vec.as_mut_ptr().add(vec.len()).write(x.into_cloned());
            vec.set_len(vec.len() + 1);
        });
    }
}

impl<T, E: Expression> Apply<T> for E {
    type Output<F: FnMut(Self::Item) -> T> = Map<E, F>;
    type ZippedWith<I: IntoExpression, F: FnMut((Self::Item, I::Item)) -> T> =
        Map<Zip<Self, I::IntoExpr>, F>;

    fn apply<F: FnMut(Self::Item) -> T>(self, f: F) -> Self::Output<F> {
        self.map(f)
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self::ZippedWith<I, F>
    where
        F: FnMut((Self::Item, I::Item)) -> T,
    {
        self.zip(expr).map(f)
    }
}

impl<E: Expression> IntoExpression for E {
    type Shape = E::Shape;
    type IntoExpr = E;

    fn into_expr(self) -> Self {
        self
    }
}
