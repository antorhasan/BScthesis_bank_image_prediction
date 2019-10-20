var geometry = 
    /* color: #d63000 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[89.53221288089765, 24.46252977918879],
          [89.53221288089765, 23.751792090040485],
          [89.90986791019452, 23.751792090040485],
          [89.90986791019452, 24.46252977918879]]], null, false);

//export pipeline for monthly median images

function maskL8sr(image) {
  // Bits 3 and 5 are cloud shadow and cloud, respectively.
  var cloudShadowBitMask = (1 << 3);
  var cloudsBitMask = (1 << 5);
  // Get the pixel QA band.
  var qa = image.select('pixel_qa');
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
                 .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  return image.updateMask(mask);
}

var cloudMaskL457 = function(image) {
  var qa = image.select('pixel_qa');
  // If the cloud bit (5) is set and the cloud confidence (7) is high
  // or the cloud shadow bit is set (3), then it's a bad pixel.
  var cloud = qa.bitwiseAnd(1 << 5)
                  .and(qa.bitwiseAnd(1 << 7))
                  .or(qa.bitwiseAnd(1 << 3));
  // Remove edge pixels that don't occur in all bands
  var mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()).updateMask(mask2);
};


//var col = collection.toList(4)
//var img = ee.Image(collection.toList(2).get(1)).select(['B2','B3','B4','B5',]);
//var img1 = collection.toList(2).get(1)
//print(collection.toList(2).get(1))
//print(img)


//filter image collection based on region of interest, date. Then apply cloud
//masking algorithm along with specific channel selection. output is a collection

for (var j = 13; j < 20; j++ ) {
  var col = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
                    .filterBounds(geometry)
                    .filterDate('20'.concat(j.toString(),'-01-01'),'20'.concat(j.toString(),'-01-31'))
                    //.map(cloudMaskL457)
                    .select(['B2','B3','B4','B5','B6','B7']);
  
  print(col);
                    
  var colList = col.toList(5000);
  var n = colList.size().getInfo();
  
  for (var i = 0; i < n; i++) {
    var img = ee.Image(colList.get(i));
    var id = img.id().getInfo();
    
    Export.image.toDrive({
      image:img,
      description: id,
      folder: 'lan8c',
      fileNamePrefix: id,
      region: geometry,
      scale: 30,
      maxPixels: 1e10})
  }

}
