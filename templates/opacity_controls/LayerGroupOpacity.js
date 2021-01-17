//https://github.com/lizardtechblog/Leaflet.OpacityControls/issues/6

//Declare global variables
var opacity_layer;
var opacity_layer_group; //ADD THIS global variable

//THEN ADD THIS control extension:

L.Control.opacitySliderGroup = L.Control.extend({
    options: {
        position: 'topright'
    },
    setOpacityLayerGroup: function (layerGroup) {
            opacity_layer_group = layerGroup;
    },
    onAdd: function (map) {
        var opacity_slider_div = L.DomUtil.create('div', 'opacity_slider_control');
        
        $(opacity_slider_div).slider({
          orientation: "vertical",
          range: "min",
          min: 0,
          max: 100,
          value: 60,
          step: 10,
          start: function ( event, ui) {
            //When moving the slider, disable panning.
            map.dragging.disable();
            map.once('mousedown', function (e) { 
              map.dragging.enable();
            });
          },
          slide: function ( event, ui ) {
            var slider_value = ui.value / 100;
            _setOpacityToLayerGroup(opacity_layer_group, slider_value);
          }
        });
        
        return opacity_slider_div;
    }
});

//ADD LOCAL FUNCTION:

function _setOpacityToLayerGroup(layerGroup, sliderValue) {
     var index = 0;
     var layersInGroup = layerGroup.getLayers();
        for (index = 0; index < layersInGroup.length; ++index) {
            layersInGroup[index].setOpacity(sliderValue);
        }        
}

//FINALLY INIT & CALL from outside as:

var multiLayers = [layer1, layer2, layer3];
var layerGroup = L.layerGroup(multiLayers);
var opacitySliderGroup = new L.Control.opacitySliderGroup();
map.addControl(opacitySliderGroup);
opacitySliderGroup.setOpacityLayerGroup(layerGroup);