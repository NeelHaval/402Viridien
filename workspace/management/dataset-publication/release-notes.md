Private dataset snapshot from the dataset review work.

- Synthetic: 82 pages; 69 ready and 13 blocked.
- Human crop subset: 27 training pages and 458 approved lithology boxes.
- Outside references: 136 labelled windows and 130 unlabelled diagnostic pages.
- Permanently sealed images and labels: excluded.

Download DATASET.json, README.md, SHA256SUMS.txt and the required ZIP files. The manifest uses relative paths and pins every source file by SHA-256. Held-out inputs and held-out reference labels are separate downloads.

The current synthetic selection blocks five empty Gulf pages and eight MC765 pages with AI-derived base-page boundaries. Earlier 77-page notes are superseded. Score only ready records and apply ignore masks. Synthetic and programmatic labels are not human-confirmed truth. Keep held-out references out of model inputs. No new inference was run.

Source data reuse licences remain unverified. Repository access remains private.
