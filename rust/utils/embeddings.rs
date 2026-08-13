use anyhow::{anyhow, Context, Result};
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use rayon::prelude::*;
use tch::{Kind, Tensor};

use crate::search::load::{get_device, PyLoadedIndex};
use crate::search::search::decompress_residuals;
use crate::utils::errors::anyhow_to_pyerr;

/// Trait that abstracts over a lazily-accessible collection of per-document
/// embedding tensors.  The Rust index-creation code calls only these methods,
/// so implementations are free to back the collection with anything: an
/// in-memory list, memory-mapped files, a remote store, etc.
pub trait Embeddings {
    /// Total number of documents.
    fn len(&self) -> usize;

    /// Whether the collection is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieve the embedding tensor for a single document.
    ///
    /// Returns a 2-D tensor of shape `(num_tokens, embedding_dim)`.
    fn get(&self, index: usize) -> Result<PyTensor>;

    /// Retrieve and concatenate embeddings for a batch of documents along
    /// the first dimension.
    ///
    /// The default implementation calls [`Self::get`] for each index and
    /// concatenates with [`tch::Tensor::cat`].  Implementations can override
    /// this for more efficient batch loading (e.g. single disk read).
    fn get_batch(&self, indices: &[usize]) -> Result<Tensor> {
        let tensors: Vec<Tensor> = indices
            .iter()
            .map(|&i| self.get(i).map(|pt| pt.0))
            .collect::<Result<_>>()?;
        if tensors.is_empty() {
            return Err(anyhow!("get_batch called with empty indices"));
        }
        Ok(Tensor::cat(&tensors, 0))
    }
}

/// Wraps any Python object that exposes `__len__`, `__getitem__`, and
/// (optionally) `get_batch` and makes it usable as an [`Embeddings`] value
/// on the Rust side.
///
/// The Python object must satisfy the `fast_plaid.Embeddings` protocol:
/// * `__len__(self) -> int`
/// * `__getitem__(self, index: int) -> torch.Tensor`
/// * `get_batch(self, indices: list[int]) -> torch.Tensor`  *(optional)*
pub struct PyEmbeddings {
    obj: PyObject,
}

impl PyEmbeddings {
    pub fn new(obj: PyObject) -> Self {
        Self { obj }
    }
}

impl Embeddings for PyEmbeddings {
    fn len(&self) -> usize {
        Python::with_gil(|py| {
            self.obj
                .bind(py)
                .len()
                .expect("Embeddings object must implement __len__")
        })
    }

    fn get(&self, index: usize) -> Result<PyTensor> {
        Python::with_gil(|py| {
            let result = self
                .obj
                .bind(py)
                .get_item(index)
                .with_context(|| format!("Embeddings.__getitem__({index}) failed"))?;
            result
                .extract::<PyTensor>()
                .with_context(|| format!("Embeddings.__getitem__({index}) did not return a Tensor"))
        })
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Tensor> {
        Python::with_gil(|py| {
            let bound = self.obj.bind(py);
            // If the object exposes get_batch, use it for efficiency.
            // Otherwise fall back to individual get() calls (trait default).
            if bound.hasattr("get_batch").unwrap_or(false) {
                let py_indices: Vec<usize> = indices.to_vec();
                let result = bound
                    .call_method1("get_batch", (py_indices,))
                    .with_context(|| "Embeddings.get_batch() failed")?;
                let py_tensor = result
                    .extract::<PyTensor>()
                    .with_context(|| "Embeddings.get_batch() did not return a Tensor")?;
                Ok(py_tensor.0)
            } else {
                let tensors: Vec<Tensor> = indices
                    .iter()
                    .map(|&i| self.get(i).map(|pt| pt.0))
                    .collect::<Result<_>>()?;
                if tensors.is_empty() {
                    return Err(anyhow!("get_batch called with empty indices"));
                }
                Ok(Tensor::cat(&tensors, 0))
            }
        })
    }
}

#[pyfunction]
pub fn reconstruct_embeddings(
    py: Python<'_>,
    index: &PyLoadedIndex,
    subset: Vec<i64>,
    device: String,
) -> PyResult<Vec<PyTensor>> {
    let device = get_device(&device)?;
    let inner = &index.inner;

    let tensors: Vec<Tensor> = py
        .allow_threads(move || {
            subset
                .into_par_iter()
                .map(|doc_id| {
                    let centroids = &inner.codec.centroids;
                    let bucket_weights = inner
                        .codec
                        .bucket_weights
                        .as_ref()
                        .ok_or_else(|| anyhow!("Index is missing bucket weights"))?;
                    let bucket_weight_indices_lookup = inner
                        .codec
                        .bucket_weight_indices_lookup
                        .as_ref()
                        .ok_or_else(|| anyhow!("Index is missing bucket weight indices lookup"))?;
                    let byte_reversed_bits_map = &inner.codec.byte_reversed_bits_map;
                    let embedding_dim = centroids.size()[1];

                    let id_tensor = Tensor::from_slice(&[doc_id]).to_device(device);
                    let (doc_codes, _) = inner.doc_codes_strided.lookup(&id_tensor, device);

                    if doc_codes.size()[0] == 0 {
                        return Ok(Tensor::empty(&[0, embedding_dim], (Kind::Float, device)));
                    }

                    let (doc_residuals, _) = inner.doc_residuals_strided.lookup(&id_tensor, device);

                    let reconstructed = decompress_residuals(
                        &doc_residuals,
                        bucket_weights,
                        byte_reversed_bits_map,
                        bucket_weight_indices_lookup,
                        &doc_codes,
                        centroids,
                        embedding_dim,
                        inner.nbits,
                    );

                    Ok(reconstructed.to_kind(Kind::Float))
                })
                .collect::<Result<Vec<Tensor>, anyhow::Error>>()
        })
        .map_err(anyhow_to_pyerr)?;

    let output_list = tensors.into_iter().map(PyTensor).collect();

    Ok(output_list)
}
