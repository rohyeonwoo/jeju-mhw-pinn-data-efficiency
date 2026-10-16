## src/mhw/datasets.py:44
```python
  44: 
  45: def build_lat_lon_2d(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
  46:     lat = ds["latitude"].values.astype(np.float32)
  47:     lon = ds["longitude"].values.astype(np.float32)
  48:     lon2d, lat2d = np.meshgrid(lon, lat)
  49:     return lat2d, lon2d
  50: 
  51: 
  52: def load_time_grid_array(ds: xr.Dataset, var_name: str) -> np.ndarray:
  53:     """Load a variable as [time, latitude, longitude], repeating 2D grids over time."""
  54:     if var_name not in ds:
  55:         raise KeyError(f"Dataset variable not found: {var_name}")
  56:     arr = ds[var_name]
  57:     if "time" in arr.dims:
  58:         spatial_dims = [dim for dim in arr.dims if dim != "time"]
  59:         if len(spatial_dims) != 2:
  60:             raise ValueError(f"Expected {var_name} to have two spatial dims, got {arr.dims}")
  61:         return arr.transpose("time", *spatial_dims).values.astype(np.float32)
  62:     if len(arr.dims) == 2:
```

## src/mhw/datasets.py:120
```python
 120: def make_time_splits(time_size: int, history: int, lead: int) -> SplitIndices:
 121:     sample_count = compute_sample_count(time_size=time_size, history=history, lead=lead)
 122:     train_end = int(sample_count * 0.7)
 123:     val_end = train_end + int(sample_count * 0.15)
 124:     all_indices = np.arange(sample_count, dtype=np.int64)
 125:     return SplitIndices(
 126:         train=all_indices[:train_end],
 127:         val=all_indices[train_end:val_end],
 128:         test=all_indices[val_end:],
 129:     )
 130: 
 131: 
```

## src/mhw/datasets.py:246
```python
 246:         self.target_array = ds["sst_mean"].values.astype(np.float32)
 247:         lat2d, lon2d = build_lat_lon_2d(ds)
 248:         self.static_arrays = {
 249:             "ocean_mask_static": self.ocean_mask,
 250:             "lat2d": lat2d.astype(np.float32),
 251:             "lon2d": lon2d.astype(np.float32),
 252:         }
 253:         self.lat2d = lat2d.astype(np.float32)
 254:         self.lon2d = lon2d.astype(np.float32)
```

## src/mhw/datasets.py:270
```python
 270:     def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
 271:         start_idx = int(self.indices[item])
 272:         input_end = start_idx + self.history
 273:         target_idx = start_idx + self.history - 1 + self.lead
 274: 
 275:         dynamic_seq = []
 276:         for name in self.dynamic_vars:
 277:             values = self.dynamic_arrays[name][start_idx:input_end]
 278:             dynamic_seq.append(self._normalize(values, name))
 279:         x_dynamic = np.stack(dynamic_seq, axis=1)
 280: 
 281:         static_seq = []
 282:         for name in self.static_vars:
 283:             static_map = self._normalize(self.static_arrays[name], name)
 284:             static_seq.append(np.repeat(static_map[None, ...], self.history, axis=0))
 285:         x_static = np.stack(static_seq, axis=1) if static_seq else None
 286: 
 287:         x = x_dynamic if x_static is None else np.concatenate([x_dynamic, x_static], axis=1)
 288: 
 289:         if self.residual_maps is not None:
 290:             residual_seq = self._normalize(self.residual_maps[start_idx:input_end], "residual")
 291:             f_seq = residual_seq[:, None, ...]
 292:         else:
 293:             f_seq = np.zeros((self.history, 1, self.ocean_mask.shape[0], self.ocean_mask.shape[1]), dtype=np.float32)
 294: 
 295:         target = self.target_array[target_idx][None, ...]
 296:         target_norm = self._normalize(target, "target")
 297:         mask = self.ocean_mask[None, ...]
 298:         target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
 299:         target_norm = np.nan_to_num(target_norm, nan=0.0, posinf=0.0, neginf=0.0)
 300: 
 301:         current_idx = input_end - 1
```

## src/mhw/jeju_hilam_dataset.py:87
```python
  87:     def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
  88:         sample = self.base[idx]
  89:         x_aug = torch.cat([sample["x"], sample["f_seq"]], dim=1)
  90:         x_dyn = x_aug[:, self.dynamic_channel_indices, :, :]
  91:         time_steps, channels, height, width = x_dyn.shape
  92:         x_nodes = x_dyn.reshape(time_steps, channels, height * width)[:, :, self._ocean_flat]
  93:         x_nodes = x_nodes.permute(2, 0, 1).contiguous()
  94: 
  95:         y_nodes = sample["y"].reshape(-1)[self._ocean_flat].contiguous()
  96:         y_raw_nodes = sample["y_raw"].reshape(-1)[self._ocean_flat].contiguous()
  97:         tm_raw_nodes = sample["Tm"].reshape(-1)[self._ocean_flat].contiguous()
  98:         tm_norm_nodes = ((tm_raw_nodes - self.target_mean) / self.target_std).contiguous()
  99: 
 100:         start_index = int(self.base.indices[idx])
 101:         current_index = start_index + self.history - 1
 102:         target_index = current_index + self.lead_days
 103: 
 104:         out = {
 105:             "x_nodes": x_nodes.float(),
 106:             "y_nodes": y_nodes.float(),
 107:             "y_raw_nodes": y_raw_nodes.float(),
 108:             "tm_norm_nodes": tm_norm_nodes.float(),
 109:             "tm_raw_nodes": tm_raw_nodes.float(),
 110:             "y_raw": sample["y_raw"].float(),
 111:             "mask": sample["mask"].float(),
 112:             "Tm": sample["Tm"].float(),
 113:             "y_raw_grid": sample["y_raw"].float(),
 114:             "mask_grid": sample["mask"].float(),
 115:             "start_index": torch.tensor(start_index, dtype=torch.long),
 116:             "current_index": torch.tensor(current_index, dtype=torch.long),
 117:             "target_index": torch.tensor(target_index, dtype=torch.long),
 118:         }
 119:         for key in (
 120:             "qnet_seq",
 121:             "hm_seq",
```

## src/mhw/jeju_hilam_graph.py:99
```python
  99: def _as_2d_lat_lon(ds) -> tuple[np.ndarray, np.ndarray]:
 100:     lat_name = "lat" if "lat" in ds else "latitude"
 101:     lon_name = "lon" if "lon" in ds else "longitude"
 102:     lat = np.asarray(ds[lat_name].values, dtype=np.float32)
 103:     lon = np.asarray(ds[lon_name].values, dtype=np.float32)
 104:     if lat.ndim == 1 and lon.ndim == 1:
 105:         lon2d, lat2d = np.meshgrid(lon, lat)
 106:         return lat2d.astype(np.float32), lon2d.astype(np.float32)
 107:     if lat.ndim == 2 and lon.ndim == 2:
 108:         return lat.astype(np.float32), lon.astype(np.float32)
 109:     raise ValueError(f"Unsupported lat/lon dimensions: lat={lat.shape}, lon={lon.shape}")
 110: 
 111: 
 112: def ocean_mask_from_dataset(ds) -> np.ndarray:
 113:     """Return the static ocean mask used by the existing Jeju dataset."""
 114: 
 115:     if "ocean_mask_static" not in ds:
 116:         raise KeyError("Dataset is missing ocean_mask_static")
 117:     mask = np.asarray(ds["ocean_mask_static"].values)
 118:     if mask.ndim != 2:
```

## src/mhw/jeju_hilam_graph.py:134
```python
 134: def _build_mapping(ocean_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
 135:     h, w = ocean_mask.shape
 136:     ocean_flat = np.flatnonzero(ocean_mask.reshape(-1)).astype(np.int64)
 137:     node_to_grid = np.column_stack(np.unravel_index(ocean_flat, (h, w))).astype(np.int64)
 138:     grid_to_node = np.full((h, w), -1, dtype=np.int64)
 139:     grid_to_node.reshape(-1)[ocean_flat] = np.arange(ocean_flat.size, dtype=np.int64)
 140:     return grid_to_node, ocean_flat, node_to_grid
```

## src/mhw/jeju_hilam_graph.py:465
```python
 465: def nodes_to_grid(
 466:     node_values: np.ndarray,
 467:     graph: JejuHiLAMGraph,
 468:     fill_value: float = 0.0,
 469: ) -> np.ndarray:
 470:     """Convert ocean node values to full grid arrays.
 471: 
 472:     Accepts either [N] or [B, N].  Returns [H, W] or [B, 1, H, W].
 473:     """
 474: 
 475:     arr = np.asarray(node_values)
 476:     if arr.ndim == 1:
 477:         out = np.full((graph.height * graph.width,), fill_value, dtype=arr.dtype)
 478:         out[graph.ocean_flat_indices] = arr
 479:         return out.reshape(graph.height, graph.width)
 480:     if arr.ndim == 2:
 481:         out = np.full((arr.shape[0], graph.height * graph.width), fill_value, dtype=arr.dtype)
 482:         out[:, graph.ocean_flat_indices] = arr
 483:         return out.reshape(arr.shape[0], 1, graph.height, graph.width)
 484:     raise ValueError(f"Expected node values with 1 or 2 dims, got {arr.shape}")
```

## src/train_jeju_hilam.py:414
```python
 414: def collect_predictions(
 415:     model: JejuHiLAM,
 416:     loader: DataLoader,
 417:     graph,
 418:     device: torch.device,
 419:     target_mean: float,
 420:     target_std: float,
 421:     precision: str,
 422:     physics_args: argparse.Namespace | None = None,
 423: ) -> dict[str, np.ndarray]:
 424:     model.eval()
 425:     pred_grids: list[np.ndarray] = []
 426:     target_grids: list[np.ndarray] = []
 427:     mask_grids: list[np.ndarray] = []
 428:     we_grids: list[np.ndarray] = []
 429:     rres_grids: list[np.ndarray] = []
 430:     target_indices: list[np.ndarray] = []
 431:     autocast = (
 432:         torch.autocast(device_type=device.type, dtype=torch.bfloat16)
 433:         if precision == "bf16" and device.type == "cuda"
 434:         else nullcontext()
 435:     )
 436: 
 437:     with torch.no_grad():
 438:         for batch in loader:
 439:             x_nodes = batch["x_nodes"].to(device, non_blocking=True)
 440:             base_nodes = batch["tm_norm_nodes"].to(device, non_blocking=True)
 441:             with autocast:
 442:                 output = model(x_nodes, base_nodes=base_nodes, return_aux=physics_args is not None)
 443:             pred_norm = output["pred"] if physics_args is not None else output
 444:             pred_raw_nodes = (pred_norm.float().cpu().numpy() * target_std) + target_mean
 445:             pred_grids.append(nodes_to_grid(pred_raw_nodes, graph, fill_value=0.0))
 446:             if physics_args is not None:
 447:                 we_grids.append(nodes_to_grid(output["we"].float().cpu().numpy(), graph, fill_value=0.0))
 448:                 rres_grids.append(nodes_to_grid(output["rres"].float().cpu().numpy(), graph, fill_value=0.0))
 449:             target_grids.append(batch["y_raw_grid"].numpy())
 450:             mask_grids.append(batch["mask_grid"].numpy())
 451:             target_indices.append(batch["target_index"].numpy())
 452: 
 453:     out = {
 454:         "pred_raw": np.concatenate(pred_grids, axis=0).astype(np.float32),
 455:         "target_raw": np.concatenate(target_grids, axis=0).astype(np.float32),
 456:         "mask": np.concatenate(mask_grids, axis=0).astype(np.float32),
 457:         "target_index": np.concatenate(target_indices, axis=0).astype(np.int64),
 458:     }
 459:     if physics_args is not None:
```
