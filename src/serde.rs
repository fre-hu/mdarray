use std::fmt::{self, Formatter};
use std::marker::PhantomData;

use serde::de::{Error, SeqAccess, Unexpected, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::buffer::Buffer;
use crate::dim::Dim;
use crate::format::Format;
use crate::grid::{DenseGrid, GridBase};
use crate::layout::Layout;
use crate::order::Order;
use crate::span::SpanBase;

struct GridVisitor<T, D: Dim, O: Order> {
    phantom: PhantomData<(T, D, O)>,
}

impl<'a, T: Deserialize<'a>, D: Dim, O: Order> Visitor<'a> for GridVisitor<T, D, O>
where
    DenseGrid<T, D::Lower, O>: Deserialize<'a>,
{
    type Value = DenseGrid<T, D, O>;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        write!(formatter, "an array of rank {}", D::RANK)
    }

    fn visit_seq<A: SeqAccess<'a>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        assert!(D::RANK > 0, "invalid rank");

        if D::RANK == 1 {
            if let Some(value) = seq.next_element()? {
                let mut vec = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                let mut shape = D::Shape::default();

                vec.push(value);

                while let Some(value) = seq.next_element()? {
                    vec.push(value);
                }

                shape[0] = vec.len();

                Ok(DenseGrid::from(vec).into_shape(shape))
            } else {
                Ok(DenseGrid::new())
            }
        } else if let Some(value) = seq.next_element::<DenseGrid<T, D::Lower, O>>()? {
            if value.is_empty() {
                Err(A::Error::invalid_length(value.len(), &self))
            } else {
                let capacity = value.len() * seq.size_hint().unwrap_or(0);
                let shape = value.shape();

                let mut grid = DenseGrid::with_capacity(capacity);
                let mut larger = D::Shape::default();

                larger[grid.dims(..D::RANK - 1)].copy_from_slice(&shape[..]);
                larger[grid.dim(D::RANK - 1)] = 1;

                grid.append(&mut value.into_shape(larger));

                while let Some(value) = seq.next_element::<DenseGrid<T, D::Lower, O>>()? {
                    if value.shape()[..] != shape[..] {
                        return Err(A::Error::invalid_value(Unexpected::Seq, &self));
                    }

                    grid.append(&mut value.into_shape(larger));
                }

                Ok(grid)
            }
        } else {
            Ok(DenseGrid::new())
        }
    }
}

impl<'a, T: Deserialize<'a>, D: Dim, O: Order> Deserialize<'a> for DenseGrid<T, D, O> {
    fn deserialize<S: Deserializer<'a>>(deserializer: S) -> Result<Self, S::Error> {
        let visitor = GridVisitor { phantom: PhantomData };

        deserializer.deserialize_seq(visitor)
    }
}

impl<B: Buffer> Serialize for GridBase<B>
where
    SpanBase<B::Item, B::Layout>: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_span().serialize(serializer)
    }
}

impl<T: Serialize, D: Dim, F: Format, O: Order> Serialize for SpanBase<T, Layout<D, F, O>> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.rank() == 0 {
            self[D::Shape::default()].serialize(serializer)
        } else if self.is_empty() {
            serializer.serialize_seq(Some(0))?.end()
        } else {
            let mut seq = serializer.serialize_seq(Some(self.size(self.dim(self.rank() - 1))))?;

            if self.rank() == 1 {
                for x in self.flatten().iter() {
                    seq.serialize_element(x)?;
                }
            } else {
                for x in self.outer_iter() {
                    seq.serialize_element(&x)?;
                }
            }

            seq.end()
        }
    }
}
