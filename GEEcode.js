// TORONTO GREEN SPACE LOSS & URBAN EXPANSION PREDICTION (2018-2030)

// 1. UI & HELPER FUNCTIONS

function legend(palette, values, names){
  var legendPanel = ui.Panel({
    style: { position: 'bottom-left', padding: '8px 15px' }
  });
  var legendTitle = ui.Label({
    value: 'Legend',
    style: { fontWeight: 'bold', fontSize: '16px', margin: '0 0 4px 0', padding: '0' }
  });
  legendPanel.add(legendTitle);
  palette.map(function(color, index){
    var colorBox = ui.Panel({
      widgets: [
        ui.Label('', { backgroundColor: color, width: '30px', height: '20px', padding: '0', margin: '0' }),
        ui.Label(names[index], { margin: '0 0 0 8px', fontSize: '14px' })
      ],
      layout: ui.Panel.Layout.flow('horizontal'),
      style: {margin: '4px 0'}
    });
    legendPanel.add(colorBox);
  });
  return legendPanel;
}

// Compute spectral indices (NDBI, NDVI, MNDWI)
var addIndices = function(image) {
  var ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI');
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI');
  return image.addBands(ndbi).addBands(ndvi).addBands(mndwi);
};

// Mask Sentinel-2 clouds using SCL band
function maskS2clouds(image) {
  var scl = image.select('SCL');
  var cloudMask = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10)).or(scl.eq(11)).not();
  return image.updateMask(cloudMask).divide(10000);
}


// 2. CONFIGURATION

var YEAR_2018 = 2018;
var YEAR_2023 = 2023;
var FUTURE_YEAR = 2030;
var bandsToSelect = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'SCL'];

// Greater Toronto Area - urban core + Greenbelt + suburban fringe
var roi = ee.Geometry.Rectangle([-80.0, 43.4, -78.8, 44.2]);

// Training ROI - Mississauga/Brampton corridor (highest growth zone)
var training_roi = ee.Geometry.Rectangle([-79.8, 43.5, -79.3, 43.9]);

var srtm = ee.Image("USGS/SRTMGL1_003");


// 3. IMAGE COLLECTION PROCESSING

print('Processing satellite imagery...');

var CLOUD_TOLERANCE = 40;

var collection2018 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(YEAR_2018 + '-06-01', YEAR_2018 + '-09-30')
    .filterBounds(roi)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_TOLERANCE))
    .select(bandsToSelect)
    .map(maskS2clouds)
    .median();
collection2018 = addIndices(collection2018);

var collection2023 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(YEAR_2023 + '-06-01', YEAR_2023 + '-09-30')
    .filterBounds(roi)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_TOLERANCE))
    .select(bandsToSelect)
    .map(maskS2clouds)
    .median();
collection2023 = addIndices(collection2023);

print('Data acquisition complete.');


// 4. LULC CLASSIFICATION & CHANGE DETECTION
var NDBI_THRESHOLD = 0.05;
var VEGETATION_THRESHOLD = 0.25;
var WATER_THRESHOLD = -0.1;
var SLOPE_THRESHOLD = 8;

var slope = ee.Terrain.slope(srtm);

function classifyBuilt(image) {
  var ndbi = image.select('NDBI');
  var ndvi = image.select('NDVI');
  var mndwi = image.select('MNDWI');
  var b11 = image.select('B11');

  return ndbi.gt(NDBI_THRESHOLD)
    .and(ndvi.lt(VEGETATION_THRESHOLD))
    .and(mndwi.lt(WATER_THRESHOLD))
    .and(b11.gt(0.05))
    .and(slope.lt(SLOPE_THRESHOLD))
    .multiply(1)
    .toByte()
    .rename('LULC');
}

var built2018 = classifyBuilt(collection2018);
var rawBuilt2023 = classifyBuilt(collection2023);

// Enforce cumulative urban growth: If it was built in 2018, it stays built in 2023.
var built2023 = rawBuilt2023.or(built2018);

// GREEN SPACE LOSS DETECTION
// Areas that WERE vegetated in 2018 but are NOW urban in 2023
var greenLoss = built2018.eq(0)
  .and(collection2018.select('NDVI').gt(0.3))
  .and(built2023.eq(1))
  .selfMask()
  .rename('GreenSpaceLost');

print('Green Space Loss layer created');

// Generate transition map
var changeMap = ee.Image(0).toByte();
var classes = [0, 1];
classes.map(function(value1, index1){
  classes.map(function(value2, index2){
    var changeValue = value1 * 1e2 + value2;
    changeMap = changeMap.where(
      built2018.eq(value1).and(built2023.eq(value2)),
      changeValue
    );
  });
});
changeMap = changeMap.rename('transition');


// 5. RANDOM FOREST TRAINING

print('Training Random Forest model...');

var variables = ee.Image([
  built2018.rename('start_lulc'),
  built2023.rename('end_lulc'),
  changeMap,
  srtm.clip(roi).rename('elevation'),
  collection2018.select('NDBI').rename('start_ndbi'),
  collection2018.select('NDVI').rename('start_ndvi'),
  collection2018.select('MNDWI').rename('start_mndwi')
]);

var propNames = ['elevation', 'start_ndbi', 'start_ndvi', 'start_mndwi'];
var predictName = 'end_lulc';

var sample = variables.stratifiedSample({
  numPoints: 5000,
  classBand: 'transition',
  scale: 150,
  region: training_roi,
  tileScale: 16
}).randomColumn();

var train = sample.filter(ee.Filter.lte('random', 0.8));
var test = sample.filter(ee.Filter.gt('random', 0.8));

print(ee.String('Training samples: ').cat(train.size().format()));
print(ee.String('Testing samples: ').cat(test.size().format()));

var model = ee.Classifier.smileRandomForest(50).train(train, predictName, propNames);

var cm = test.classify(model, 'prediction').errorMatrix('end_lulc', 'prediction');
print('Model Performance Summary:');
print(cm);
print(ee.String('Accuracy: ').cat(cm.accuracy().format('%.2f')));
print(ee.String('Kappa Coefficient: ').cat(cm.kappa().format('%.3f')));


// 6. VALIDATION

print('------------------------------------------------');
print('Validating model on 2023 data...');

var predicted2023 = variables.classify(model);
var mndwi_val = collection2023.select('MNDWI');

var predicted2023Clean = predicted2023
    .and(mndwi_val.lt(WATER_THRESHOLD))
    .toByte();

var match = predicted2023Clean.eq(built2023).rename('match');

var spatialAccuracy = match.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: training_roi,
  scale: 150,
  maxPixels: 1e13,
  tileScale: 16,
  bestEffort: true
}).get('match');

print(ee.String('Correctly Predicted Pixels: ').cat(ee.Number(spatialAccuracy).multiply(100).format('%.2f')).cat(' %'));
print('------------------------------------------------');


// 7. 2030 PREDICTION

var variablesFuture = ee.Image([
  built2023.rename('start_lulc'),
  changeMap,
  srtm.rename('elevation'),
  collection2023.select('NDBI').rename('start_ndbi'),
  collection2023.select('NDVI').rename('start_ndvi'),
  collection2023.select('MNDWI').rename('start_mndwi')
]);

var rawPrediction = variablesFuture.classify(model, 'LULC_Prediction');

var mndwi_2023 = collection2023.select('MNDWI');

// Create a 1km realistic expansion buffer around the current city limits
// Create a realistic 250-meter expansion buffer around the current city limits
var expansionZone = built2023.focal_max(250, 'circle', 'meters');
var landscapeConstraints = mndwi_2023.lt(WATER_THRESHOLD)
    .and(slope.lt(SLOPE_THRESHOLD))
    .and(expansionZone.eq(1)); // Forces the AI to only build near existing infrastructure

var constrainedPredictions = rawPrediction.and(landscapeConstraints);
var lulcFuture = constrainedPredictions.or(built2023).toByte().rename('LULC_Prediction');


// 8. AREA CALCULATIONS & CHARTS

function calculateArea(image, className) {
  var areaImage = ee.Image.pixelArea().divide(10000);
  var area = areaImage.updateMask(image.eq(1))
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: roi,
      scale: 150,
      maxPixels: 1e13,
      tileScale: 16,
      bestEffort: true
    });
  return ee.Number(area.get('area'));
}

var area2018 = calculateArea(built2018, 'Built 2018');
var area2023 = calculateArea(built2023, 'Built 2023');
var area2030 = calculateArea(lulcFuture, 'Built 2030');

print('Area Calculation (Hectares):');
print(ee.String('Built Area 2018: ').cat(area2018.format('%.2f')));
print(ee.String('Built Area 2023: ').cat(area2023.format('%.2f')));
print(ee.String('Predicted Area 2030: ').cat(area2030.format('%.2f')));

var lulcListArea = [
  { year: 2018, image: built2018 },
  { year: 2023, image: built2023 },
  { year: 2030, image: lulcFuture }
];

var lulcAreafeatures = ee.FeatureCollection(lulcListArea.map(function(dict){
  var imageArea = ee.Image.pixelArea().divide(10000);
  var reduceArea = imageArea.addBands(dict.image).reduceRegion({
    reducer: ee.Reducer.sum().setOutputs(['area']).group(1, 'class'),
    scale: 150,
    geometry: roi,
    bestEffort: true,
    tileScale: 16
  }).get('groups');

  var features = ee.FeatureCollection(ee.List(reduceArea).map(function(dictionary){
    dictionary = ee.Dictionary(dictionary);
    var label = ee.Number(dictionary.get('class')).eq(1) ? 'Built' : 'Not Built';
    dictionary = dictionary.set('year', ee.Number(dict.year).toInt());
    dictionary = dictionary.set('LULC', label);
    return ee.Feature(null, dictionary);
  }));
  return features;
})).flatten();

var chartArea = ui.Chart.feature.groups(lulcAreafeatures, 'year', 'area', 'LULC')
  .setChartType('ColumnChart')
  .setOptions({
    title: 'Greater Toronto Area Urban Expansion (2018-2030)',
    hAxis: {title: 'Year', format: '####'},
    vAxis: {title: 'Built Area (Hectares)'},
    colors: ['#808080', '#FF0000'],
    series: { 0: {color: '#A0A0A0'}, 1: {color: '#FF0000'} }
  });
print(chartArea);


// 9. EXPORTS

var newGrowth = lulcFuture.subtract(built2023).selfMask();

Export.image.toDrive({
  image: collection2023.select(['B4', 'B3', 'B2']).addBands(built2023.rename('label')).toFloat(),
  description: 'Toronto_Urban_2023',
  folder: 'GreenWatch_Toronto',
  region: roi,
  scale: 150,
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: lulcFuture,
  description: 'Toronto_Urban_2030_Prediction',
  folder: 'GreenWatch_Toronto',
  region: roi,
  scale: 150,
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: newGrowth,
  description: 'Toronto_New_Growth_2023_2030',
  folder: 'GreenWatch_Toronto',
  region: roi,
  scale: 150,
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: built2018,
  description: 'Toronto_Urban_2018',
  folder: 'GreenWatch_Toronto',
  region: roi,
  scale: 150,
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: greenLoss,
  description: 'Toronto_GreenSpace_Lost_2018_2023',
  folder: 'GreenWatch_Toronto',
  region: roi,
  scale: 150,
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13
});


// 10. VISUALIZATION

var builtPalette = ['000000', 'FF0000'];
var visParams = {min: 0, max: 1, palette: builtPalette};
var ndbiVis = {min: -0.2, max: 0.5, palette: ['008000', 'FFFFFF', 'FF0000']};
var ndviVis = {min: -0.2, max: 0.8, palette: ['8B4513', 'FFFF00', '00FF00']};
var mndwiVis = {min: -0.5, max: 0.5, palette: ['4B0082', '8A2BE2', '0000FF']};
var transitionVis = {min: 1, max: 111, palette: ['00FF00', 'FFFF00', 'FF0000']};

var leftMap = ui.Map();
leftMap.setOptions('HYBRID');
leftMap.addLayer(predicted2023Clean, {min: 0, max: 1, palette: ['black', 'orange']}, 'Validation: Predicted 2023 (Orange)', false);
leftMap.addLayer(changeMap, transitionVis, 'Built Area Transition (2018-2023)', false);
leftMap.addLayer(greenLoss, {min: 0, max: 1, palette: ['FF0000']}, 'Green Space Lost (2018-2023)', true);
leftMap.addLayer(collection2023.select('NDBI'), ndbiVis, 'NDBI 2023', false);
leftMap.addLayer(collection2023.select('NDVI'), ndviVis, 'NDVI 2023', false);
leftMap.addLayer(collection2023.select('MNDWI'), mndwiVis, 'MNDWI 2023', false);
leftMap.addLayer(built2023, visParams, 'Built Area 2023', true);
leftMap.add(ui.Label('Toronto 2023 (Current)', {position: 'top-center', fontSize: '16px', fontWeight: 'bold', backgroundColor: 'rgba(255, 255, 255, 0.8)'}));

var rightMap = ui.Map();
rightMap.setOptions('HYBRID');
rightMap.addLayer(collection2023.select('NDBI'), ndbiVis, 'NDBI 2023', false);
rightMap.addLayer(collection2023.select('NDVI'), ndviVis, 'NDVI 2023', false);
rightMap.addLayer(collection2023.select('MNDWI'), mndwiVis, 'MNDWI 2023', false);
rightMap.addLayer(newGrowth, {min: 0, max: 1, palette: ['FFFF00']}, 'New Growth 2023-2030', false);
rightMap.addLayer(lulcFuture, visParams, 'Predicted Built 2030', true);
rightMap.add(ui.Label('Toronto 2030 (Predicted)', {position: 'top-center', fontSize: '16px', fontWeight: 'bold', backgroundColor: 'rgba(255, 255, 255, 0.8)'}));


// 11. GROWTH HIGHLIGHTS

var newGrowthStats = newGrowth.multiply(ee.Image.pixelArea().divide(10000))
  .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: roi,
    scale: 150,
    maxPixels: 1e13,
    tileScale: 16,
    bestEffort: true
  });

var newGrowthAreaVal = newGrowthStats.get('LULC_Prediction');
print(ee.String('New Development (2023->2030): ')
  .cat(ee.Number(newGrowthAreaVal).format('%.2f'))
  .cat(' hectares'));


// 12. LEGENDS & SPLIT PANEL

var legendLeft = legend(
  ['000000', 'FF0000', 'FF6600'],
  [0, 1, 2],
  ['Not Built', 'Built 2023', 'Green Space Lost']
);
var legendRight = legend(
  ['000000', 'FF0000', 'FFFF00'],
  [0, 1, 2],
  ['Not Built', 'Predicted Built 2030', 'New Growth']
);
leftMap.add(legendLeft);
rightMap.add(legendRight);

var splitPanel = ui.SplitPanel({
  firstPanel: leftMap,
  secondPanel: rightMap,
  wipe: true,
  style: {stretch: 'both'}
});

var linker = ui.Map.Linker([leftMap, rightMap]);
leftMap.centerObject(roi, 10);

ui.root.clear();
ui.root.add(splitPanel);
