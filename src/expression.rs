#[cfg(feature = "nightly")]
use std::alloc::Allocator;

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::array::GridArray;
use crate::dim::Dim;
use crate::expr::{Cloned, Copied, Enumerate, Map, Zip};
use crate::iter::Iter;
use crate::traits::{Apply, IntoCloned, IntoExpression};

/// Expression trait, for multidimensional iteration.
pub trait Expression: IntoIterator {
    /// Array dimension type.
    type Dim: Dim;

    #[doc(hidden)]
    const IS_REPEATABLE: bool;

    #[doc(hidden)]
    const SPLIT_MASK: usize;

    /// Returns the shape of the array.
    fn shape(&self) -> <Self::Dim as Dim>::Shape;

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

    /// Creates an expression which gives tuples of the current index and the element.
    fn enumerate(self) -> Enumerate<Self, <Self::Dim as Dim>::Shape>
    where
        Self: Sized,
    {
        Enumerate::new(self)
    }

    /// Evaluates the expression into a new array.
    fn eval(self) -> GridArray<Self::Item, Self::Dim>
    where
        Self: Sized,
    {
        GridArray::from_expr(self)
    }

    /// Evaluates the expression into a new array with the specified allocator.
    #[cfg(feature = "nightly")]
    fn eval_in<A: Allocator>(self, alloc: A) -> GridArray<Self::Item, Self::Dim, A>
    where
        Self: Sized,
    {
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
    fn eval_into<D: Dim, A: Allocator>(
        self,
        grid: &mut GridArray<Self::Item, D, A>,
    ) -> &mut GridArray<Self::Item, D, A>
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
        self.len() == 0
    }

    /// Returns the number of elements in the array.
    fn len(&self) -> usize {
        self.shape()[..].iter().product()
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
        Self::Dim::RANK
    }

    /// Returns the number of elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    fn size(&self, dim: usize) -> usize {
        assert!(dim < Self::Dim::RANK, "invalid dimension");

        self.shape()[dim]
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
    unsafe fn reset_dim(&mut self, dim: usize, count: usize);

    #[doc(hidden)]
    unsafe fn step_dim(&mut self, dim: usize);

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
    type Output<F: FnMut(Self::Item) -> T> =
        Map<impl Expression<Item = Self::Item, Dim = E::Dim>, F>;

    type ZippedWith<I: IntoExpression, F: FnMut((Self::Item, I::Item)) -> T> =
        Map<impl Expression<Item = (Self::Item, I::Item), Dim = <E::Dim as Dim>::Max<I::Dim>>, F>;

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
    type Dim = E::Dim;
    type IntoExpr = E;

    fn into_expr(self) -> Self {
        self
    }
}
