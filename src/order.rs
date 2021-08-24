/// Ordering for indexing and iteration over array elements.
#[derive(PartialEq, Eq)]
pub enum Order {
    /// Row-major ordering, for indexing and iteration over array elements.
    ColumnMajor,

    /// Column-major ordering, for indexing and iteration over array elements.
    RowMajor,
}
