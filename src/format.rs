use std::fmt::Debug;
use std::iter::FusedIterator;
use std::slice::{Iter, IterMut};

use crate::dim::Dim;
use crate::iter::{LinearIter, LinearIterMut};
use crate::mapping::{DenseMapping, GeneralMapping, LinearMapping, Mapping, StridedMapping};
use crate::order::Order;

/// Array format trait for memory layout.
pub trait Format: Copy + Debug + Default {
    /// Corresponding format which may have non-uniform stride.
    type NonUniform: NonUniform;

    /// Corresponding format which may have non-unit inner stride.
    type NonUnitStrided: NonUnitStrided;

    /// Corresponding format with uniform stride.
    type Uniform: Uniform;

    /// Corresponding format with unit inner stride.
    type UnitStrided: UnitStrided;

    /// Array iterator type.
    type Iter<'a, T: 'a>: Clone
        + DoubleEndedIterator
        + ExactSizeIterator
        + FusedIterator
        + Iterator<Item = &'a T>;

    /// Mutable array iterator type.
    type IterMut<'a, T: 'a>: DoubleEndedIterator
        + ExactSizeIterator
        + FusedIterator
        + Iterator<Item = &'a mut T>;

    #[doc(hidden)]
    type Mapping<D: Dim, O: Order>: Mapping<D, Self, O>;
}

/// Trait for format types which may have non-uniform stride.
pub trait NonUniform: Format {}

/// Trait for format types which may have non-unit inner stride.
pub trait NonUnitStrided: Format {}

/// Trait for format types with uniform stride.
pub trait Uniform: Format {}

/// Trait for format types with unit inner stride.
pub trait UnitStrided: Format {}

/// Dense array format type.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dense;

/// General array format type.
#[derive(Clone, Copy, Debug, Default)]
pub struct General;

/// Linear array format type.
#[derive(Clone, Copy, Debug, Default)]
pub struct Linear;

/// Strided array format type.
#[derive(Clone, Copy, Debug, Default)]
pub struct Strided;

impl Format for Dense {
    type NonUniform = General;
    type NonUnitStrided = Linear;
    type Uniform = Self;
    type UnitStrided = Self;

    type Iter<'a, T: 'a> = Iter<'a, T>;
    type IterMut<'a, T: 'a> = IterMut<'a, T>;

    type Mapping<D: Dim, O: Order> = DenseMapping<D, O>;
}

impl Format for General {
    type NonUniform = Self;
    type NonUnitStrided = Strided;
    type Uniform = Dense;
    type UnitStrided = Self;

    type Iter<'a, T: 'a> = Iter<'a, T>;
    type IterMut<'a, T: 'a> = IterMut<'a, T>;

    type Mapping<D: Dim, O: Order> = GeneralMapping<D, O>;
}

impl Format for Linear {
    type NonUniform = Strided;
    type NonUnitStrided = Self;
    type Uniform = Self;
    type UnitStrided = Dense;

    type Iter<'a, T: 'a> = LinearIter<'a, T>;
    type IterMut<'a, T: 'a> = LinearIterMut<'a, T>;

    type Mapping<D: Dim, O: Order> = LinearMapping<D, O>;
}

impl Format for Strided {
    type NonUniform = Self;
    type NonUnitStrided = Self;
    type Uniform = Linear;
    type UnitStrided = General;

    type Iter<'a, T: 'a> = LinearIter<'a, T>;
    type IterMut<'a, T: 'a> = LinearIterMut<'a, T>;

    type Mapping<D: Dim, O: Order> = StridedMapping<D, O>;
}

impl NonUniform for General {}
impl NonUniform for Strided {}

impl NonUnitStrided for Linear {}
impl NonUnitStrided for Strided {}

impl Uniform for Dense {}
impl Uniform for Linear {}

impl UnitStrided for Dense {}
impl UnitStrided for General {}
