function _check_error(is_error, model) {
  if (is_error) {
    model.alert_id = 2;
    model.trigger_alert = true;
  }
}

function sync_south(source, lines_cds, rects_cds, model) {
  if (model.is_move) return;

  var y0 = model.ys + source.value;
  var [is_error, y0] = _update_south(y0, lines_cds, rects_cds, model);

  _check_error(is_error, model);
  source.value = y0 - model.ys;
}

function sync_east(source, lines_cds, rects_cds, model) {
  if (model.is_move) return;

  var x1 = model.xe - source.value;
  var [is_error, x1] = _update_east(x1, lines_cds, rects_cds, model);

  _check_error(is_error, model);
  source.value = model.xe - x1;
}

function sync_north(source, lines_cds, rects_cds, model) {
  if (model.is_move) return;

  var y1 = model.ye - source.value;
  var [is_error, y1] = _update_north(y1, lines_cds, rects_cds, model);

  _check_error(is_error, model);
  source.value = model.ye - y1;
}

function sync_west(source, lines_cds, rects_cds, model) {
  if (model.is_move) return;

  var x0 = model.xs + source.value;
  var [is_error, x0] = _update_west(x0, lines_cds, rects_cds, model);

  _check_error(is_error, model);
  source.value = x0 - model.xs;
}

function update_plot(event, lines_cds, rects_cds, model) {
  // Resetting server side alert
  if (model.alert_id > 0) model.alert_msg = "";

  if (event.event_name == "tap") {
    var idxs = lines_cds.selected.indices;
    var n = idxs.length;

    if (n > 1) {
      idxs = [idxs[n - 1]];
      lines_cds.selected.indices = idxs;
    }

    model.is_move = n > 0;

    lines_cds.change.emit();
    rects_cds.change.emit();
    //returnd
  }

  if (!model.is_move) return;

  const i = lines_cds.selected.indices[0];
  var is_error = false;
  switch (i) {
    case 0:
      [is_error, y0] = _update_south(event.y, lines_cds, rects_cds, model);
      break;
    case 1:
      [is_error, x1] = _update_east(event.x, lines_cds, rects_cds, model);
      break;
    case 2:
      [is_error, y1] = _update_north(event.y, lines_cds, rects_cds, model);
      break;
    case 3:
      [is_error, x0] = _update_west(event.x, lines_cds, rects_cds, model);
      break;
  }

  _check_error(is_error, model);
  if (is_error) {
    model.is_move = false;
    lines_cds.selected.indices = [];
    rects_cds.selected.indices = [];
  }
}

function _update_south(y0, lines_cds, rects_cds, model) {
  // Checking y position is in grid
  var is_error = y0 >= model.y1;
  if (is_error) {
    model.alert_msg =
      "Moving South sponge boundary past the North spong boundary.";
    y0 = model.y1 - model.dy;
  } else if (y0 < model.ys) {
    model.alert_msg = "Moving South sponge boundary outside domain.";
    y0 = model.ys;
    is_error = true;
  }

  // Rounding to nearest partition points
  const j0 = Math.round((y0 - model.ys) / model.dy);
  if (j0 == model.j0) return [is_error, y0];
  y0 = j0 * model.dy + model.ys;

  model.j0 = j0;
  model.y0 = y0;

  lines_cds.data.ys[0] = [model.y0, model.y0];
  lines_cds.data.ys[1] = [model.y0, model.y1];
  lines_cds.data.ys[3] = [model.y0, model.y1];

  rects_cds.data.top[0] = model.y0;

  lines_cds.change.emit();
  rects_cds.change.emit();

  return [is_error, y0];
}

function _update_east(x1, lines_cds, rects_cds, model) {
  var is_error = x1 <= model.x0;
  if (is_error) {
    model.alert_msg =
      "Moving East sponge boundary past the West spong boundary.";
    x1 = model.x0 + model.dx;
  } else if (x1 > model.xe) {
    model.alert_msg = "Moving East sponge boundary outside domain.";
    x1 = model.xe;
    is_error = true;
  }

  const i1 = Math.round((x1 - model.xs) / model.dx);
  if (i1 == model.i1) return [is_error, x1];
  x1 = i1 * model.dx + model.xs;

  model.i1 = i1;
  model.x1 = x1;

  lines_cds.data.xs[1] = [model.x1, model.x1];
  lines_cds.data.xs[0] = [model.x0, model.x1];
  lines_cds.data.xs[2] = [model.x0, model.x1];

  rects_cds.data.left[1] = model.x1;
  rects_cds.data.right[0] = model.x1;
  rects_cds.data.right[2] = model.x1;

  lines_cds.change.emit();
  rects_cds.change.emit();

  return [is_error, x1];
}

function _update_north(y1, lines_cds, rects_cds, model) {
  var is_error = y1 <= model.y0;
  if (is_error) {
    model.alert_msg =
      "Moving North sponge boundary past the South sponge boundary.";
    y1 = model.y0 + model.dy;
  } else if (y1 > model.ye) {
    model.alert_msg = "Moving North sponge boundary outside domain.";
    y1 = model.ye;
    is_error = true;
  }

  const j1 = Math.round((y1 - model.ys) / model.dy);
  if (j1 == model.j1) return [is_error, y1];
  y1 = j1 * model.dy + model.ys;

  model.j1 = j1;
  model.y1 = y1;

  lines_cds.data.ys[2] = [model.y1, model.y1];
  lines_cds.data.ys[1] = [model.y0, model.y1];
  lines_cds.data.ys[3] = [model.y0, model.y1];

  rects_cds.data.bottom[2] = model.y1;

  lines_cds.change.emit();
  rects_cds.change.emit();

  return [is_error, y1];
}

function _update_west(x0, lines_cds, rects_cds, model) {
  var is_error = x0 >= model.x1;
  if (is_error) {
    model.alert_msg =
      "Moving West sponge boundary past the East sponge boundary.";
    x0 = model.x1 - model.dx;
  } else if (x0 < model.xs) {
    model.alert_msg = "Moving West sponge boundary outside domain.";
    x0 = model.xs;
    is_error = true;
  }

  const i0 = Math.round((x0 - model.xs) / model.dx);
  if (i0 == model.i0) return [is_error, x0];
  x0 = i0 * model.dx + model.xs;

  model.i0 = i0;
  model.x0 = x0;

  lines_cds.data.xs[3] = [model.x0, model.x0];
  lines_cds.data.xs[0] = [model.x0, model.x1];
  lines_cds.data.xs[2] = [model.x0, model.x1];

  rects_cds.data.right[3] = model.x0;
  rects_cds.data.left[0] = model.x0;
  rects_cds.data.left[2] = model.x0;

  lines_cds.change.emit();
  rects_cds.change.emit();

  return [is_error, x0];
}
