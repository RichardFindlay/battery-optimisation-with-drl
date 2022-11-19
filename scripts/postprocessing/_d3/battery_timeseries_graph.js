
function timeseries_graph(file, battery_capacity, battery_power) {
  console.log("ts graph loaded") }

//  intial dates to show on graph
var filter_date = new Date(2018, 0, 1)
var end_date = new Date(2018, 0, 8) 

// create js calender picker
$("#calendar-tomorrow").flatpickr({
    mode: 'single',
    maxDate: "2018-12-25",
    minDate: "2017-1-1",
    defaultDate: "2018-1-1",
    onChange: function(dates) {

            //  remove existing data
            // plot.selectAll('rect').remove()
            // svg_100.select('.location_2').remove()

            filter_date = new Date(dates);
            end_date = new Date(dates)

            if (current_width <= 625) {
              end_date.setDate(end_date.getDate() + 3);
            } else {
              end_date.setDate(end_date.getDate() + 7);
            }

            newDataDatePath = dataset.filter(function(d) {
              return d.date >= filter_date && d.date < end_date;
            })

            // remove existing bars on bar chart
            plot.selectAll(".location").remove()
            plot.selectAll(".location_2").remove()

            svg_Xdates_noon.selectAll(".tick").remove()

            // update domains
             x_bar.domain(newDataDatePath.map(function(d) { return d.id; })).padding(0.2);
             x_ref.domain([d3.min(newDataDatePath, function(d) { return +d.id; }), d3.max(newDataDatePath, function(d) { return +d.id; })])
             x_input.domain([d3.min(newDataDatePath, function(d) { return +d.id; }), d3.max(newDataDatePath, function(d) { return +d.id; })])
             x.domain([d3.min(newDataDatePath, function(d) { return +d.id; }), d3.max(newDataDatePath, function(d) { return +d.id; })])
             x_dates.domain(d3.extent(newDataDatePath, function(d){return d.date;}));

             xAxis_dates_hours.scale(x_dates)

            // pass filtered data to graphs
            draw_price_line(dataset, startDate = filter_date, endDate = end_date)
            draw_bar_soc(dataset, startDate = filter_date, endDate = end_date)
            draw_bar_act(dataset, startDate = filter_date, endDate = end_date)
            
            svg_100.select('.line').attr("d", input_line(newDataDatePath))
            svg_100.select('.subpath').attr("d", input_line(newDataDatePath))

            svg.select(".domain").remove(); 

            // reset slider to zero
            handle.attr("cx", x(0))

            dataset_update = newDataDatePath

            // make sure path is drawn in full
            update_offset(1000, resize=true)


            // update tick text for dates
            svg_Xdates_noon.call(xAxis_dates_noon)
            svg.select(".x.axis_dates_main").call(xAxis_dates_days_main_divide).select(".domain").remove();

            // re-format tick text - slider dates
            svg_Xdates_noon.selectAll('.tick text').each(function(_,i){
              if(i%2==0){
                d3.select(this).remove()
              }
            })

            svg_Xdates_noon.selectAll('.tick text')
                .style("text-transform", "uppercase")
                .attr("fill","grey")
                .attr("transform", "translate(0," + (-30) + ")")

            if (current_width <= 625) {
            d3.select(".x.axis_dates_main").selectAll('.tick text')
                .style("text-transform", "uppercase")
                .attr("opacity", 0.6)
                .attr("fill", "#293241")
                .attr("text-anchor", "center")
                .attr("font-size", "14")
                .attr("transform", "translate(" + current_width/8 + "," + (-height+20) + ")")
                .select(".domain").remove();
            } else {
            d3.select(".x.axis_dates_main").selectAll('.tick text')
                .style("text-transform", "uppercase")
                .attr("opacity", 0.6)
                .attr("fill", "#293241")
                .attr("text-anchor", "center")
                .attr("font-size", "14")
                .attr("transform", "translate(" + current_width/14 + "," + (-height+20) + ")")
                .select(".domain").remove();
            }

            //  remove intial tick for presentation
            d3.select('.x.axis_dates_main_ticks .tick line:first-child').remove()

            // update profit text
            charg_dis_prices = newDataDatePath.map(function(d) {return d.price_data * d.action * battery_power})
            profit = charg_dis_prices.reduce(getSum, 0.0).toFixed(2)
            profit_label.text("Profit 💰:" + formatter.format(profit))


        }
    })




// Adapted from: https://bl.ocks.org/officeofjane/47d2b0bfeecfcb41d2212d06d095c763
var formatDateIntoYear = d3.timeFormat("%Y");
var formatDate = d3.timeFormat("%b %Y");
var parseDate = d3.timeParse("%m/%d/%y");

var startDate = new Date("2004-11-01"),
    endDate = new Date("2017-04-01");

current_width = parseInt(d3.select('#test').style('width'), 10) 

if (current_width <= 625) {
  width = parseInt(d3.select('#test').style('width'), 10) - 85
  var margin = {top:10, right:25, bottom:0, left:85}
  height = 200 - margin.top - margin.bottom;
} else {
  width = parseInt(d3.select('#test').style('width'), 10) 
  var margin = {top:10, right:25, bottom:0, left:50}
  height = 240 - margin.top - margin.bottom;
}

//  some intial params
const intial_width = width
var aspectRatio= '16:9';
var viewBox = '0 0 ' + aspectRatio.split(':').join(' ');

// declare main graph
var svg = d3.select("#vis")
    .append("svg")
    .attr("width", width + margin.right + margin.left)
    .attr("height", 50)  
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")")

// global vars for project
var x_input,y_input,
    svg_input_Xaxis,
    svg_input_Xaxis_date,
    svg_input_Yaxis,
    dataset,
    dataset_update,
    height,
    margin,
    length,
    x_bar,
    x_ref,
    x,
    filter_date, 
    end_date,
    newDataDate,
    profit_label,
    svg_Xdates,
    svg_Xdates_noon,
    svg_Xdates_days,
    svg_Xdates_days_main,
    second_txt,
    newDataDatePath,
    second_txt,
    sec_text

// input line graph
var svg_100 = d3.select("#my_dataviz100")
    .append("svg")
    .attr("width", width + margin.right + margin.left + 50) 
    .attr("height", height + margin.bottom + margin.top + 0)  
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")")

searchcolumn = d3.select(".container_")
                    .style("width", (width + margin.left)+ 'px')
                    .style("height", 0 + 'px');

////// INPUT LINE SETUP //////////

x_input = d3.scaleLinear()
  .range([0, width])
  .clamp(true);

x_input_date = d3.scaleTime()
  .range([0, width])

y_input = d3.scaleLinear()
  .domain([0, 100])
  .range([height, 0]);

xAxis = d3.axisBottom().scale(x_input).tickSizeOuter(0).tickValues([]);
yAxis = d3.axisLeft().scale(y_input).tickSizeOuter(0);

xAxis_dates_main= d3.axisBottom().scale(x_input_date).tickSizeOuter(0).tickValues([]);

input_line = d3.line()
                .curve(d3.curveBasis)
                .x(function(d) { return x_input(d.id) })
                .y(function(d) { return y_input(d.price_data) })

////////// slider //////////
var moving = false;
var currentValue = 0;
var targetValue = width;
var dataset;
var dataset_update;
var group = svg_100.append("g");

////// SOC BAR SETUP //////////

var x = d3.scaleLinear()
   // .domain([0, 168])
    .range([0, width])
    .clamp(true);

var x_ref = d3.scaleLinear()
    // .domain([0, 168])
    .range([0, width])
    .clamp(true);

var x_bar = d3.scaleBand()
  .range([0, width])

var y_bar = d3.scaleLinear()
  .domain([-1, 1])
  .range([height, 0]);

var y_bar_2 = d3.scaleLinear()
  .domain([-1, 1])
  .range([height, 0]);

xAxis_bar_soc = d3.axisBottom().scale(x_bar).tickSizeOuter(0).tickValues([]);
yAxis_bar_soc = d3.axisRight().scale(y_bar).tickSizeOuter(0);

x_dates = d3.scaleTime()
  .range([0, width])

xAxis_dates_hours = d3.axisBottom().scale(x_dates).tickSizeOuter(0).tickFormat(d3.timeFormat("")).ticks(d3.timeHour.every(1)).tickSize(5);
xAxis_dates_noon = d3.axisBottom().scale(x_dates).tickSizeOuter(0).tickFormat(d3.timeFormat("%d/%m")).ticks(d3.timeHour.every(12)).tickSize(10);
xAxis_dates_days = d3.axisBottom().scale(x_dates).tickSizeOuter(0).tickFormat(d3.timeFormat("")).ticks(d3.timeHour.every(24)).tickSize(18.5);

xAxis_dates_days_main_divide = d3.axisBottom().scale(x_dates).tickSizeOuter(0).tickFormat(d3.timeFormat("%A")).ticks(d3.timeHour.every(24)).tickSize(height);

xAxis_dates_days_main_divide_ticks = d3.axisBottom().scale(x_dates).tickSizeOuter(0).tickFormat(d3.timeFormat("%A")).ticks(d3.timeHour.every(24)).tickSize(height);

// add y-axis gridlines
svg_100.selectAll("line.horizontalGrid").data(y_input.ticks(4)).enter()
    .append("line")
        .attr("class", "horizontalGrid")
        .attr("x1", 0)
        .attr("x2", width)
        .attr("y1", function(d){ return y_input(d) + 23.5;})
        .attr("y2", function(d){ return y_input(d) + 23.5;})
        .attr("fill", "none")
        .attr("shape-rendering", "crispEdges")
        .attr("stroke", "grey")
        .attr("stroke-width", "0.2px")
        .attr("opacity", 0.3)
        // .style("stroke-dasharray", 4)
        .attr("z-index", -1) 

// add y-axis labels
svg_100.data(y_input.ticks(7)).enter()
    .append("text")
        .attr("class", "verticaltext_days")
        .attr("opacity", 1.0)
        .attr("transform", "translate(" + 100 + "," + 0 + ")")

var slider = svg.append("g")
    .attr("class", "slider")
    .attr("transform", "translate(" + 0 + "," + 10 + ")");

slider.append("line")
    .attr("class", "track")
    .attr("x1", x_input.range()[0])
    .attr("x2", x_input.range()[1])
  .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
    .attr("class", "track-inset")
  .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
    .attr("class", "track-overlay")
    .call(d3.drag()
        // .on("start.interrupt", function() { slider.interrupt(); })
        .on("start drag", function() {
          currentValue = d3.mouse(this)[0]

          update_offset(x_ref.invert(currentValue))
          
          // offset = x.invert(currentValue)
          update(x_ref.invert(currentValue))
          soc_bar_update(x_ref.invert(currentValue))

          info_line(currentValue)

          mousemoved(currentValue);
        })   
    )
    .on("click", function() {
      currentValue = d3.mouse(this)[0]
      update_offset(x_ref.invert(currentValue))
      update(x_ref.invert(currentValue));
      soc_bar_update(x_ref.invert(currentValue));
      
      info_line(currentValue);
      mousemoved(currentValue);
    })
    .on("touchstart", function() {
      currentValue = d3.mouse(this)[0]
      update_offset(x_ref.invert(currentValue))
      update(x_ref.invert(currentValue));
      soc_bar_update(x_ref.invert(currentValue));
      
      info_line(currentValue);
      mousemoved(currentValue);
    })

var handle = slider.insert("circle", ".track-overlay")
    .attr("transform", "translate(" + 0 + "," + 0 + ")")
    .attr("class", "handle")
    .attr("r", 9);

var info_line_text = svg_100.append("g").append("text")  
    .attr("class", "info_line_text")
    .attr("font-size", "10")
    .style("fill", '#293241')
    
var info_line_text2 = svg_100.append("g").append("text")  
    .attr("class", "info_line_text1")
    .attr("font-size", "10")
    .style("fill", '#f79256')
    // .attr("font-weight","bold")

var info_line_text3 = svg_100.append("g").append("text")  
    .attr("class", "info_line_text2")
    .attr("font-size", "10")
    .style("fill", '#457b9d')
    // .attr("font-weight","bold")


var info_line_time = svg_100.append("g").append("text")  
    .attr("class", "info_line_text3")
    .attr("font-size", "16")
    .style("fill", '#293241')
    // .attr("font-weight","bold")


// cumulative profit for the plotted week
var subPathData = [
    []
  ];

  updateSubPath(subPathData);

var bisect = d3.bisector(function(d) { return d.price_data; }).left;

function getSum(total, num) {
  return total + num;
}

// reload data
function reload_data() {


d3.csv(file, prepare, function(data) {
  //  reload and filter for specified date range


  datanew = data.filter(function(row) {
      return row['date'] < datefil;
  })

  dataset = data;

})
}

// interactivity func
function mousemoved(current_coords) {

    mouse_coord = {
        "id": current_coords,
        // "price_data": current_coords[1]
    }
  
    subPathData = [
        []
    ];
    
    for (var i = 0; i < dataset_update.length; i++) {
        var coord_id = dataset_update[i]['id'];
        var coord_price_data = dataset_update[i]['price_data'];

        if ((x_input(coord_id) <= mouse_coord.id)) {
            subPathData[0].push({'id': coord_id, 'price_data': coord_price_data});
        }
    }

    pathLength = d3.select(".line").node().getTotalLength()

      var x = mouse_coord.id; 
      var beginning = x, end = pathLength, target;
  while (true) {
    target = Math.floor((beginning + end) / 2);
    pos = d3.select(".line").node().getPointAtLength(target);
    if ((target === end || target === beginning) && pos.x !== x) {
        break;
    }
    if (pos.x > x)      end = target;
    else if (pos.x < x) beginning = target;
    else                break; //position found
  }

    mouse_coord.id = String(x_input.invert(mouse_coord.id))
    mouse_coord.price_data = String(y_input.invert(pos.y))
    subPathData[0].push(mouse_coord);

    updateSubPath(subPathData);
    // Calculate the length of the subpath - this will be line length
    // up to the point we're mouseover-ing
    // subpath_length = d3.select("subpath")[0][0].getTotalLength();
    subpath_length = d3.select(".subpath").node().getTotalLength();

};

var mouseG = svg_100
  .append("g")
  .attr("class", "mouse-over-effects");

mouseG
  .append("path") // this is the black vertical line to follow mouse
  .attr("class", "mouse-line")
  // .style("stroke", "#393B45") //6E7889
  .style("stroke", "#293241")
  .style("stroke-width", "1.0px")
  .style("opacity", 0.75)

// func for interactivity and to show data as slider is moved
function info_line(cur_pos=null) {
  if (cur_pos === null) {
    x_ = d3.event.layerX ;
  } else {
    x_ = cur_pos
  }

    pathLength = d3.select(".line").node().getTotalLength()

    var x = x_ ; 
    var beginning = x, end = pathLength, target;

    while (true) {
      target = Math.floor((beginning + end) / 2);
      pos = d3.select(".line").node().getPointAtLength(target);
      if ((target === end || target === beginning) && pos.x !== x) {
          break;
      }
      if (pos.x > x)      end = target;
      else if (pos.x < x) beginning = target;
      else                break; //position found
    }

    bar_data = [
        []
    ];

    for (var i = 0; i < dataset_update.length; i++) {
        var coord_id = dataset_update[i]['id'];
        var coord_price_data1 = dataset_update[i]['action'] * 10; // 10MW battery power (baseline)
        var coord_price_data2 = dataset_update[i]['soc'] * 20; // 20MWh battery capacity (baseline)
        var coord_price_data3 = dataset_update[i]['date'];
        var coord_price_data4 = dataset_update[i]['price_data'];

        if ((x_input(coord_id) <= x_ + 5)) {
            bar_data.push({'id': coord_id, 'action': coord_price_data1, 'soc': coord_price_data2, 'date': d3.timeFormat("%H:%M %p")(coord_price_data3), 'price_data': coord_price_data4});
        }
    }

          point_input
            .attr('cx', pos.x)
            .attr('cy', pos.y)
            .style("opacity", "1");

          d3.select("#my_dataviz100")
            .select(".mouse-line")
            .attr("d", function() {
                var d = "M" + (pos.x) + "," + height;
                d += " " + (pos.x) + "," + 0;
              return d;
            })
      
          d3.select("#my_dataviz100")
            .select(".mouse-line")
            .style("opacity", "1")

          info_line_text
            .text('Price: ' + formatter.format(bar_data[bar_data.length-1]['price_data']))
            .attr('x', pos.x + 5)
            .attr('y', 8)
            .style("opacity", "1")


          info_line_text2
            .text('Action: ' + parseFloat(bar_data[bar_data.length-1]['action']).toFixed(2) + " MW")
            .attr('x', pos.x + 5)
            .attr('y', 19)
            .style("opacity", "1")

          info_line_text3
            .text('SoC: ' + parseFloat(bar_data[bar_data.length-1]['soc']).toFixed(2) + " MW")
            .attr('x', pos.x + 5)
            .attr('y', 30)
            .style("opacity", "1")


          info_line_time
            .text(bar_data[bar_data.length-1]['date'])
            .attr('x', pos.x + 5)
            .attr('y', 50)
            .style("opacity", "1")

          

          subpath_cur = d3.select(".line").node().getPointAtLength(d3.select(".subpath").node().getTotalLength());

          cur_end = (x_input.invert(subpath_cur.x)).toFixed(2)
          cur_lineinfo = x_input.invert((x_))

        }


  svg.on("mouseleave", function () {
    d3.select("#my_dataviz100")
      .select(".mouse-line")
      .style("opacity", "0")
      point_input.style("opacity", "0")
      info_line_text.style("opacity", "0")
      info_line_text2.style("opacity", "0")
      info_line_text3.style("opacity", "0")
      info_line_time.style("opacity", "0")
  })


  svg_100.on("mouseleave", function () {
    d3.select("#my_dataviz100")
      .select(".mouse-line")
      .style("opacity", "0")
      point_input.style("opacity", "0")
      info_line_text.style("opacity", "0")
      info_line_text2.style("opacity", "0")
      info_line_text3.style("opacity", "0")
      info_line_time.style("opacity", "0")
  })


function updateSubPath(subData) {
    // JOIN
    var subpathselect = group.selectAll(".subpath")
        .data(subData);
    // UPDATE
    subpathselect.attr("d", input_line);
    // ENTER
    subpathselect.enter()
        .append("path")
        .attr("fill", "none")
        .attr("fill-opacity", 0)
        .attr("stroke-width", 0)
        .attr("d", input_line)
        .attr("class", "subpath");
}

const point_input = svg_100.append('circle')
  .attr('r', 4)
  .style("fill", 'none')
  .style("stroke", '#293241')
  .style("opacity", "0");


var plot = svg_100.append("g")
    .attr("class", "plot")
    // .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

d3.csv(file, prepare, function(data) {

  current_width = parseInt(d3.select('#test').style('width'), 10) 

  if (current_width <= 625) {
    datefil = new Date(2018, 0, 4)
    x.domain([0, 72])
    x_ref.domain([0, 72])
  } else {
    datefil = new Date(2018, 0, 8)
    x.domain([0, 168])
    x_ref.domain([0, 168])
  }

  datanew = data.filter(function(row) {
      return row['date'] < datefil;
  })
  
  dataset = data;
  dataset_update = datanew;


    //  plot bars for soc
  x_bar.domain(datanew.map(function(d) { return d.id; }))
    .padding(0.2);

  // drawPlot(dataset);
  draw_price_line(datanew);
  draw_bar_soc(datanew);
  draw_bar_act(datanew);

  charg_dis_prices = datanew.map(function(d) {return d.price_data * d.action * battery_power})
  profit = charg_dis_prices.reduce(getSum, 0.0).toFixed(2)
  profit_label.text("Profit 💰:" + formatter.format(profit))

  start_date = new Date(2018, 0, 1)
  newDataDatePath_int = dataset.filter(function(d) {
  return d.date >= start_date && d.date < datefil;
  })

  // instantiate line graph plot
  var path = svg_100.append("path")
    .datum(dataset)
    .attr("class", "line")
    .attr("fill", "none")
    .attr("opacity", 0.8)
    .attr("stroke", "#444444")
    .attr("stroke-width", 1.75)
    .attr("d", input_line(newDataDatePath_int))
    // .on("mousemove", mousemoved);

  var length = path.node().getTotalLength();

  x_dates.domain(d3.extent(datanew, function(d){return d.date;}))
  x_input_date.domain(d3.extent(datanew, function(d){return d.date;}))

  svg_Xdates = svg.append("g")
    .attr("class", "x axis_dates")
    .attr("transform", "translate(0," + 10 + ")")
    .call(xAxis_dates_hours)
  
  svg_Xdates_noon = svg.append("g")
    .attr("class", "x axis_dates")
    .attr("transform", "translate(0," + 10 + ")")
    .call(xAxis_dates_noon)

  svg_Xdates_days = svg.append("g")
    .attr("class", "x axis_dates")
    .attr("transform", "translate(0," + 10 + ")")
    .call(xAxis_dates_days)

  svg_Xdates_noon.selectAll('.tick text').each(function(_,i){
    if(i%2==0){
      d3.select(this).remove()
    }
  })

  svg_Xdates_noon.selectAll('.tick text')
      .style("text-transform", "uppercase")
      .attr("fill","#BEBEBE")
      .attr("transform", "translate(0," + (-30) + ")")
      .select(".domain").remove();

  // svg_Xdates_noon.selectAll(".tick text").remove()
  svg_Xdates_days_main = svg.append("g")
    .attr("class", "x axis_dates_main")
    .call(xAxis_dates_days_main_divide)
        .attr("fill", "none")
        .attr("shape-rendering", "crispEdges")
        .attr("stroke", "grey")
        .attr("stroke-width", "0px")
        .attr("opacity", 0.2)
        // .style("stroke-dasharray", 4)
    .select(".domain").remove();

  if (current_width <= 625) {
  d3.select(".x.axis_dates_main").selectAll('.tick text')
      .style("text-transform", "uppercase")
      .attr("opacity", 0.6)
      .attr("fill", "#293241")
      .attr("text-anchor", "center")
      .attr("font-size", "14")
      .attr("transform", "translate(" + current_width/8 + "," + (-height+20) + ")")
      .select(".domain").remove();
  } else {
  d3.select(".x.axis_dates_main").selectAll('.tick text')
      .style("text-transform", "uppercase")
      .attr("opacity", 0.6)
      .attr("fill", "#293241")
      .attr("text-anchor", "center")
      .attr("font-size", "14")
      .attr("transform", "translate(" + current_width/14 + "," + (-height+20) + ")")
      .select(".domain").remove();
  }

  // primary y-axis label
  if (current_width <= 625) {
  svg_100.append("text")
      .attr("text-anchor", "start")
      .attr("y", -30)
      .attr("x", -margin.top)
      .text("Price (£/MW)")
      .attr("fill","grey")
      .attr('transform', 'translate(' + 0 + ',' + ((height / 2) + 25) + ')rotate(-90)')
      .style("font-size", "0.65em")
  } else {
  svg_100.append("text")
      .attr("text-anchor", "start")
      .attr("y", -margin.left+20)
      .attr("x", -margin.top)
      .text("Price (£/MW)")
      .attr("fill","grey")
      .attr('transform', 'translate(' + 0 + ',' + ((height / 2) + 25) + ')rotate(-90)')
      .style("font-size", "0.65em")
  }

  // primary x-axis label
  sec_text = svg_100.append("text")
      .attr("class", "secondary_yaxis")
      .attr("text-anchor", "start")
      .attr("y",width + 45)
      .attr("x", -height/1.35)
      .text("SoC & Normalised Action")
      .attr("fill","grey")
      .attr('transform', 'rotate(-90)')
      .style("font-size", "0.65em")
      // .attr('transform', 'translate(' + (width + 75) + ',' + ((height / 2) + 65) + ')rotate(-90)')

  //  remove intial tick for presentation
  d3.select('.x.axis_dates_main .tick line:first-child').remove()

  // get some of the thresholds
  svg_Xdates.lower()
  svg_Xdates_noon.lower()
  svg_Xdates_days.lower()
  svg_Xdates_days_main.lower()

  path.attr("stroke-dasharray", length + " " + length)
    .attr("stroke-dashoffset", 0);

  d3.select("#play-button")
    .on("click", function() {
    var button = d3.select(this);
    if (button.text() == "Pause") {
      moving = false;
      clearInterval(timer);
      timer = 0;
      button.text("Simulate ⚡");
    } else {
      moving = true;
      timer = setInterval(step, 75);
      button.text("Pause");
    }
  })
    // call width resize for graphs
    d3.select(window).on('resize', resize);
})

// helper func to prepare data
function prepare(d) {
  d.id = d.id;
  d.price_data = d.price_data
  d.date = d3.timeParse("%d/%m/%Y %H:%M")(d.date)
  d.action = d.action;
  return d;
}
  
function step() {
  if (currentValue < 0) {
    currentValue = 0
  }
  currentValue = currentValue + (targetValue/250);

  update_offset(x_ref.invert(currentValue))
  update(x_ref.invert(currentValue));
  soc_bar_update(x_ref.invert(currentValue));
  
  info_line(currentValue);
  mousemoved(currentValue);

  if (currentValue > targetValue) {
    moving = false;
    currentValue = 0;
    clearInterval(timer);
    playButton.text("Simulate ⚡");
  }
}

soc_Xaxis = svg_100.append("g")
  .attr("stroke", "grey")
  .attr("opacity", "0.25")
  .attr("class", "x axis_soc")
  .attr("transform", "translate(0," + height/2 + ")")
  .call(xAxis_bar_soc)

// Add Y axis
soc_Yaxis = svg_100.append("g")
  .attr("class", "y axis_soc")
  .call(yAxis_bar_soc)
  .attr("transform", "translate(" + width + "," + 0 + ")")
  .select(".domain").remove();

act_Xaxis = svg_100.append("g")
  .attr("class", "x axis_act")
  .attr("transform", "translate(0," + height/2 + ")")
  .call(xAxis_bar_soc)
  .select(".domain").remove();

svg_input_Xaxis = svg_100.append("g")
  .attr("class", "x axis")
  .attr("transform", "translate(0," + height + ")")
  .call(xAxis)
  .select(".domain").remove();

svg_input_Xaxis_date = svg_100.append("g")
  .attr("class", "x axis date")
  .attr("transform", "translate(0," + height + ")")
  .call(xAxis_dates_main)
  // .select(".domain").remove();

svg_input_Yaxis = svg_100.append("g")
  .attr("class", "y axis")
  .attr("transform", "translate(0," +  0 + ")")
  .call(yAxis)
  .select(".domain").remove();

  // format all tick marks
  svg_100.select('.y.axis_soc').selectAll('.tick text')

  // only include secondary ticks (seconday axis)
  svg_100.select('.y.axis_soc').selectAll('.tick').each(function(_,i) {
    if (i%2==0) {
      d3.select(this).remove()
    }
  })

  // only include secondary ticks (primary axis)
  svg_100.select('.y.axis').selectAll('.tick').each(function(_,i) {
    if (i%2==0) {
      d3.select(this).remove()
    }
  })

  svg_100.selectAll('.tick')
        .attr("fill", "none")
        .attr("shape-rendering", "crispEdges")
        .attr("stroke", "grey")
        .attr("stroke-width", "0.0px")
        .attr("opacity", 0.4)

function draw_price_line(line_data, startDate = new Date(2018, 0, 1), endDate = new Date(2018, 0, 8)) {

  // filtered data on date range
  var newDataDate = line_data.filter(function(d) {
    return (d.date >= startDate && d.date < endDate);
  })

  // INPUT GRAPH
  x_input.domain([d3.min(newDataDate, function(d) { return +d.id ; }), d3.max(newDataDate, function(d) { return +d.id ; })])
  // y_input.domain([0, d3.max(newDataDate, function(d) { return +d.price_data; })])

}


// add profit label - place here to ensure in front of Action bars
if (intial_width <= 625) {
var profit_label = svg_100.append("g").append("text")  
    .attr("class", "profit_label")
    .attr("font-size", "0.6em")
    // .text("Profit 💰: this is a test ")
    // .attr("transform", "translate(" + (width - 125)  + "," + (height + 20) + ")")
    profit_label.attr("transform", "translate(" + ((width /2)-25) + "," + (0) + ")")
} else {
var profit_label = svg_100.append("g").append("text")  
    .attr("class", "profit_label")
    .attr("font-size", "0.7em")
    // .text("Profit 💰: this is a test ")
    // .attr("transform", "translate(" + (width - 125)  + "," + (height + 20) + ")")
    .attr("transform", "translate(" + (width - 100) + "," + (height - 5) + ")")
}


function draw_bar_soc(bar_data, startDate = new Date(2018, 0, 1), endDate = new Date(2018, 0, 8)) {

  // filtered data on date range
  var newDataDate = bar_data.filter(function(d) {
    return (d.date >= startDate && d.date < endDate);
  })

  var locations = plot.selectAll(".location")
    .data(newDataDate);

  // if filtered dataset has more circles than already existing, transition new ones in
  locations
    .enter()
    .append("rect")
      .attr("class", "location")
      .attr("x", function(d) { return x_bar(d.id); })
      .attr("y", function(d) { return y_bar(0); })
      .attr("width", x_bar.bandwidth())
      .attr("fill", "#457b9d")
      .attr("opacity", "0.6")
      // .attr("height", function(d) { return height/2 - y_bar(d.soc); })
        .transition()
        .duration(300)
        .attr("y", function(d) { return y_bar(d.soc); })
        .attr("height", function(d) { return height/2 - y_bar(d.soc); })

  // Update
  locations
      .attr("x", function(d) { return x_bar(d.id); })
      .attr("width", x_bar.bandwidth())


    // remove previous bars fro animation
  locations.exit().remove();

}


function draw_bar_act(bar_data, startDate = new Date(2018, 0, 1), endDate = new Date(2018, 0, 8)) {

  // filtered data on date range
  var newDataDate = bar_data.filter(function(d) {
    return d.date >= startDate && d.date < endDate;
  })

  var locations_2 = plot.selectAll(".location_2")
    .data(newDataDate);

  // if filtered dataset has more circles than already existing, transition new ones in
  locations_2
    .enter()
    .append("rect")
      .attr("class", "location_2")
      .attr("x", function(d) { return x_bar(d.id); })
      .attr("y", function(d) { return y_bar(0); })   
      .attr("width", x_bar.bandwidth())
      .attr("fill", "#f79256") // 
      .attr("opacity", "0.8")
        .transition()
        .duration(300)
        .attr("y", function(d) { return y_bar(Math.max(0, d.action)); })
        .attr("height", function(d) { return Math.abs(y_bar(d.action) - y_bar(0)); })

  // Update
  locations_2
      .attr("x", function(d) { return x_bar(d.id); })
      .attr("width", x_bar.bandwidth())

  locations_2.exit().remove();

}


function update(h) {
  handle.attr("cx", x_ref(h));
}


var formatter = new Intl.NumberFormat('en-UK', {
  style: 'currency',
  currency: 'GBP',
});


function soc_bar_update(h) {

  // filter data set and redraw plot
  var newData = dataset_update.filter(function(d) {
    return d.id <= h;
  })

  // added condition for zero
  if (newData.length == 1) {
      newData = []
  }


  charg_dis_prices = newData.map(function(d) {return d.price_data * d.action * battery_power})
  profit = charg_dis_prices.reduce(getSum, 0.0).toFixed(2)
  profit_label.text("Profit 💰:" + formatter.format(profit))

  draw_bar_soc(newData, startDate = filter_date, endDate = end_date);
  draw_bar_act(newData, startDate = filter_date, endDate = end_date);

}

// create function to handle 
var resize = function(e) {

// update svg line chart /////////////////////////////////////////////////////////
start_date = filter_date

if (current_width <= 625) {
  end_date.setDate(start_date.getDate() + 3);
} else {
  end_date.setDate(start_date.getDate() + 7);
}

if (current_width <= 625) {
  width = parseInt(d3.select('#test').style('width'), 10) + 25
  var margin = {top:10, right:25, bottom:0, left:85}
} else {
  width = parseInt(d3.select('#test').style('width'), 10) + 75
  var margin = {top:10, right:25, bottom:0, left:50}
}

  newDataDatePath_int = dataset.filter(function(d) {
  return d.date >= start_date && d.date < end_date;
  })

  output_graph_width = width - margin.left - margin.right

  d3.select("#my_dataviz100").select("svg").attr("width", width+50)
  svg_100.attr("width", width)
  x_input.range([0, output_graph_width]);
  y_input.range([height, 0]);

  xAxis.scale(x_input);
  yAxis.scale(y_input);

  svg_input_Xaxis.call(xAxis)
  // svg_input_Yaxis.call(yAxis)

  svg_100.select(".x.axis").call(xAxis)
  svg_100.select('.line').attr("d", input_line(newDataDatePath_int))

  update_offset(1000, resize=true)

  // update sub-path line chart /////////////////////////////////////////////////////////
  searchcolumn = d3.select(".container_")
                    .style("width", (output_graph_width) + 'px')

  // update svg soc-bar chart /////////////////////////////////////////////////////////
  x_bar.range([0, output_graph_width])
  x_ref.range([0, output_graph_width])

  xAxis_bar_soc.scale(x_bar)
  yAxis_bar_soc.scale(y_bar)

  soc_Xaxis.call(xAxis_bar_soc)
  soc_Yaxis.call(yAxis_bar_soc)

  plot.select(".x.axis_soc").call(xAxis_bar_soc)
  plot.select(".y.axis_soc").call(yAxis_bar_soc)

  draw_price_line(newDataDatePath_int, startDate = filter_date, endDate = end_date)
  draw_bar_soc(newDataDatePath_int, startDate = filter_date, endDate = end_date)

  // update svg action-bar chart /////////////////////////////////////////////////////////
  d3.select("#button_div").attr("width", output_graph_width - 50)

  // update slider /////////////////////////////////////////////////////////
  // d3.select("#my_dataviz100").select("svg").attr("width", width)
  d3.select("#vis").select("svg").attr("width", width)

  x.range([0, output_graph_width])

  svg.select('.track').attr("x2", x.range()[1])
  svg.select('.track-overlay').attr("x2", x.range()[1])
  svg.select('.track-inset').attr("x2", x.range()[1])

  draw_bar_act(newDataDatePath_int, startDate = filter_date, endDate = end_date)

  // update tick lines on track
  x_dates.range([0, output_graph_width])

  xAxis_dates_hours.scale(x_dates)
  xAxis_dates_noon.scale(x_dates)
  xAxis_dates_days.scale(x_dates)

  svg_Xdates.call(xAxis_dates_hours)
  svg_Xdates_noon.call(xAxis_dates_noon)
  svg_Xdates_days.call(xAxis_dates_days)

  // update day name ticks
  xAxis_dates_days_main_divide.scale(x_dates)

  svg.select('.x.axis_dates_main').call(xAxis_dates_days_main_divide)

  if (intial_width <= 625) {
  d3.select(".x.axis_dates_main").selectAll('.tick text')
      .style("text-transform", "uppercase")
      .attr("opacity", 0.6)
      .attr("fill", "#293241")
      .attr("text-anchor", "center")
      .attr("font-size", "14")
      .attr("transform", "translate(" + width/8 + "," + (-height+20) + ")")
      .select(".domain").remove();
  } else {
  d3.select(".x.axis_dates_main").selectAll('.tick text')
      .style("text-transform", "uppercase")
      .attr("opacity", 0.6)
      .attr("fill", "#293241")
      .attr("text-anchor", "center")
      .attr("font-size", "14")
      .attr("transform", "translate(" + width/14 + "," + (-height+20) + ")")
      .select(".domain").remove();
  }

if (intial_width <= 625) {
  d3.select(".secondary_yaxis").attr("x", -(height/2) - 65).attr("y", output_graph_width+40) 
} else {
  d3.select(".secondary_yaxis").attr("x", -(height/2) - 65).attr("y", width-30) 
}


  if (current_width <= 625) {
    profit_label.attr("transform", "translate(" + ((output_graph_width /2)-25) + "," + (0) + ")")
  } else {
    profit_label.attr("transform", "translate(" + (output_graph_width - 100) + "," + (height - 5) + ")")
  }

  svg_100.select(".x.axis_dates_main_ticks").call(xAxis_dates_days_main_divide_ticks)

  svg_100.selectAll("line.horizontalGrid")
          .attr("x1", 0)
          .attr("x2", output_graph_width)

  svg_100.select(".y.axis_soc")
    .attr("transform", "translate(" + (output_graph_width) + "," + 0 + ")")
    .select(".domain").remove();

    svg_100.selectAll(".legend_labels").attr("x", function(d,i){ return ((output_graph_width/2)-35) + i*(size+50)})
    svg_100.selectAll(".legend_boxes").attr("x", function(d,i){ return ((output_graph_width/2)-50) + i*(size+50)})

}


function lineFiltered(data) {
  return line(data.filter(function (d) { return !!d }))
}


function update_offset(offset, resize=false) {

  var length = svg_100.select('.line').node().getTotalLength()

  if (resize === false){
    var offset_2 = svg_100.select('.subpath').node().getTotalLength()
  } else {
    var offset_2 = length
    handle.attr("cx", x(0))
  }

  svg_100.select('.line')
      .attr("stroke-dasharray", length + " " + length)
      .attr("stroke-dashoffset", length - offset_2);

} 


// create legend
var keys = ["SoC", "Action", "Price"]

var color = ['#457b9d','#f79256', "#444444"]

var color = d3.scaleOrdinal()
  .domain(keys)
  .range(color);

var size = 6
svg_100.selectAll("mydots")
  .data(keys)
  .enter()
  .append("rect")
    .attr('class', 'legend_boxes')
    .attr("x", function(d,i){ return ((width/2)-55) + i*(size+50)})
    .attr("y",  function(d, i){ if (i==2) {
      return height -9 } else {
    return height -10}})
    .attr("width",  function(d, i){ if (i==2) {
      return size +3 } else {
    return size}})
    .attr("height",  function(d, i){ if (i==2) {
      return size -3 } else {
    return size}})
    .attr("opacity", "0.6")
    .style("fill", function(d){ return color(d)})

leg_labels = svg_100.selectAll("mylabels")
  .data(keys)
  .enter()
  .append("text")
    .attr('class', 'legend_labels')
    .attr("x", function(d,i){ return ((width/2)-45) + i*(size+50)})
    .attr("y", height-6.5)
    .style("fill", function(d){ return color(d)})
    .text(function(d){ return d})
    .attr("text-anchor", "left")
    .attr('font-size', 10)
    .style("alignment-baseline", "middle")






















