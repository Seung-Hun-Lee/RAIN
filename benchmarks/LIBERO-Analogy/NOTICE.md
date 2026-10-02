# Licensing and provenance

The licenses below apply to the named dependencies and helper files. They do not establish a license for the LIBERO-Analogy task bundles or project-specific geometry and scoring helpers.

- LIBERO is an external dependency, distributed under MIT by its authors and the Hugging Face `lerobot-libero` maintainers. Its installed license is preserved in `licenses/LIBERO-MIT.txt`.
- Two unmodified helper files from Physical Intelligence's [OpenPI commit 54cbaee6ae0c010a1ed431871cdaa8f4684ac709](https://github.com/Physical-Intelligence/openpi/tree/54cbaee6ae0c010a1ed431871cdaa8f4684ac709) are vendored under `src/libero_analogy/_vendor/openpi`: `msgpack_numpy.py` and `image_tools.py`. The Apache-2.0 license is preserved in `licenses/OpenPI-Apache-2.0.txt`. The codec states that it adapts [Lev E. Givon's msgpack-numpy](https://github.com/lebedov/msgpack-numpy); its upstream BSD-3-Clause copyright/license notice is additionally preserved in `licenses/msgpack-numpy-BSD-3-Clause.md`. No model code, weights, or broader client stack are vendored. Source hashes are in `SOURCE_MANIFEST.json`.
- The 60 task bundles and project-specific geometry/scoring helpers are from LIBERO-Analogy and its evaluation runtime. This notice does not grant a license for those files.
- Simulator textures, meshes, and third-party object assets are not redistributed. `ASSET_MANIFEST.json` identifies reference content from the upstream asset repository; it is not a license grant for that content.

Keep this notice and the included third-party notices when redistributing the corresponding files. `SOURCE_MANIFEST.json` records file hashes and source provenance.
