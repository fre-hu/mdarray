use std::fmt::Debug;

/// Array format trait.
pub trait Format: Copy + Debug + Default {
    /// General or strided array format.
    type NonDense: NonDense;

    /// Dense or general array format.
    type UnitStrided: UnitStrided;
}

/// General or strided array format trait.
pub trait NonDense: Format {}

/// Dense or general array format trait.
pub trait UnitStrided: Format {}

/// Dense array format type.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dense;

/// General array format type.
#[derive(Clone, Copy, Debug, Default)]
pub struct General;

/// Strided array format type.
#[derive(Clone, Copy, Debug, Default)]
pub struct Strided;

impl Format for Dense {
    type NonDense = General;
    type UnitStrided = Dense;
}

impl Format for General {
    type NonDense = General;
    type UnitStrided = General;
}

impl Format for Strided {
    type NonDense = Strided;
    type UnitStrided = General;
}

impl NonDense for General {}
impl NonDense for Strided {}

impl UnitStrided for Dense {}
impl UnitStrided for General {}
