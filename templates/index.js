function postJSONData(JSONData, resID) {
    $.ajax({
      type: "POST",
      url: "https://git.sarlab.ca:5050/storeJSON/" + resID,
      contentType: "application/json",
      data: JSONData,
    })
      .done(function () {
        $("#serverstatus")[0].innerHTML = "SAVED";
      })
      .fail(function () {
        alert("error");
      })
      .always(function () {
        // whatever we'd do in all cases
      });
  }
  function getJSONData(resID, fGroup) {
    $.ajax({
      type: "GET",
      url: "https://cocalc.sarlab.ca:5050/readJSON/" + resID,
      dataType: "json",
    })
      .done(function (data) {
        $("#serverstatus")[0].innerHTML = "Server Data Loaded.";
        L.geoJson(data["features"], {
          onEachFeature: function (feature, layer) {
            fGroup.addLayer(layer);
          },
        });
        $("#serverstatus")[0].innerHTML = "Loaded Annotations added to Image.";
      })
      .fail(function (data) {
        if (data["status"] == 404) {
          $("#serverstatus")[0].innerHTML = "No annotations found on Server.";
        } else {
          alert("Error Loading Data");
        }
      });
  }
  
  //create the map
  var bottomMap = L.tileLayer(
    "https://pfeifer.phas.ubc.ca/map/tiles/{z}/tile_{x}_{y}.png",
    {
      layers: "BottomLayer",
      maxZoom: 7,
      minZoom: 1,
      format: "image/png",
      transparent: true,
    }
  );
  
  var topMap = new L.tileLayer("./tiles/{z}/tile_{x}_{y}.png", {
    layers: "TopLayer",
    maxZoom: 7,
    minZoom: 1,
    format: "image/png",
    transparent: true,
  });
  
  var map = new L.map("map", {
      maxZoom: 7,
      minZoom: 1,
      crs: L.CRS.Simple,
      layers: [bottomMap, topMap],
      zoomControl: true,
    }).setView([0, 0], 1),
    drawnItems = L.featureGroup().addTo(map),
    sourcelabels = L.layerGroup().addTo(map),
    histolayer = L.tileLayer(
      "https://pfeifer.phas.ubc.ca/map/tiles/{z}/tile_{x}_{y}.png",
      {
        attribution: "AIMlab",
      }
    ).addTo(map);
  
  //brightness and contrast slider, opacity slider for bottom layer
  histolayer.getContainer().style.filter =
    "brightness(100%)" + "contrast(100%)" + "opacity(100%)";
  var slider1 = document.getElementById("brightness-slider");
  var slider2 = document.getElementById("contrast-slider");
  var slider3 = document.getElementById("opacity-slider");
  
  var brightnessSpan = document.getElementById("brightness-value");
  var contrastSpan = document.getElementById("contrast-value");
  var opacitySpan = document.getElementById("opacity-value");
  
  var brightness = slider1.value;
  var contrast = slider2.value;
  var opacity = slider3.value;
  
  slider1.addEventListener("input", function (e) {
    brightness = e.target.value;
    brightnessSpan.textContent = brightness + "%";
    histolayer.getContainer().style.filter =
      "opacity(" +
      opacity +
      "%) brightness(" +
      this.value +
      "%) contrast(" +
      contrast +
      "%)";
  });
  slider2.addEventListener("input", function (e) {
    contrast = e.target.value;
    contrastSpan.textContent = contrast + "%";
    histolayer.getContainer().style.filter =
      "opacity(" +
      opacity +
      "%) brightness(" +
      brightness +
      "%) contrast(" +
      this.value +
      "%)";
  });
  slider3.addEventListener("input", function (e) {
    opacity = e.target.value;
    opacitySpan.textContent = opacity + "%";
    histolayer.getContainer().style.filter =
      "opacity(" +
      this.value +
      "%) brightness(" +
      brightness +
      "%) contrast(" +
      contrast +
      "%)";
  });
  
  //second group of sliders
  topMap.getContainer().style.filter =
    "brightness(100%)" + "contrast(100%)" + "opacity(100%)";
  var slider4 = document.getElementById("brightness-slider2");
  var slider5 = document.getElementById("contrast-slider2");
  var slider6 = document.getElementById("opacity-slider2");
  
  var brightnessSpan2 = document.getElementById("brightness-value2");
  var contrastSpan2 = document.getElementById("contrast-value2");
  var opacitySpan2 = document.getElementById("opacity-value2");
  
  var brightness2 = slider4.value;
  var contrast2 = slider5.value;
  var opacity2 = slider6.value;
  
  slider4.addEventListener("input", function (e) {
    brightness2 = e.target.value;
    brightnessSpan2.textContent = brightness2 + "%";
    topMap.getContainer().style.filter =
      "opacity(" +
      opacity2 +
      "%) brightness(" +
      this.value +
      "%) contrast(" +
      contrast2 +
      "%)";
  });
  slider5.addEventListener("input", function (e) {
    contrast2 = e.target.value;
    contrastSpan2.textContent = contrast2 + "%";
    topMap.getContainer().style.filter =
      "opacity(" +
      opacity2 +
      "%) brightness(" +
      brightness2 +
      "%) contrast(" +
      this.value +
      "%)";
  });
  slider6.addEventListener("input", function (e) {
    opacity2 = e.target.value;
    opacitySpan2.textContent = opacity2 + "%";
    topMap.getContainer().style.filter =
      "opacity(" +
      this.value +
      "%) brightness(" +
      brightness2 +
      "%) contrast(" +
      contrast2 +
      "%)";
  });
  
  //Create the opacity controls
  var higherOpacity = new L.Control.higherOpacity();
  map.addControl(higherOpacity);
  var lowerOpacity = new L.Control.lowerOpacity();
  map.addControl(lowerOpacity);
  var opacitySlider = new L.Control.opacitySlider();
  map.addControl(opacitySlider);
  //specifies layer which opacity will be modified:
  opacitySlider.setOpacityLayer(topMap);
  //set initial opacity to 0.5
  //topMap.setOpacity(0.5);
  
  var baseLayers = { Histolayer: histolayer };
  
  var overlays = {
    "Source Labels": sourcelabels,
    Annotations: drawnItems,
    "Top Layer": topMap,
  };
  L.control
    .layers({ histo: histolayer }, overlays, {
      position: "topleft",
      collapsed: true,
    })
    .addTo(map);
  map.addControl(
    new L.Control.Draw({
      edit: { featureGroup: drawnItems, poly: { allowIntersection: false } },
      draw: { polygon: { allowIntersection: false, showArea: true } },
    })
  );
  // load server data
  getJSONData("8888", drawnItems);
  
  // define event hooks to save data etc.
  map.on("draw:drawstart draw:editstart draw:deletestart", function (event) {
    $("#serverstatus")[0].innerHTML = "NOT YET SAVED";
  });
  map.on("draw:created", function (event) {
    var layer = event.layer;
    drawnItems.addLayer(layer);
    postJSONData(JSON.stringify(drawnItems.toGeoJSON()), "8888");
  });
  map.on("draw:edited draw:deleted", function (event) {
    var layer = event.layer;
    postJSONData(JSON.stringify(drawnItems.toGeoJSON()), "8888");
  });
  
  // https://gis.stackexchange.com/questions/59571/how-to-add-text-only-labels-on-leaflet-map-with-no-icon
  //opacity may be set to zero for no marker
  var marker = new L.marker([-39.5, 50], { opacity: 0.2 });
  marker.bindTooltip("some histoname", {
    permanent: true,
    className: "my-label",
    offset: [0, 0],
  });
  marker.addTo(sourcelabels);
  