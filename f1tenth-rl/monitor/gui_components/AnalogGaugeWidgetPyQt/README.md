# AnaloGaugeWidgetPyQt

Custom QWidget 
- compatible to PyQt4 and PyQt5

![AnalogGaugeWidgetDemo Image](img/AnalogGaugeWidgetDemo.JPG?raw=true "AnalogGaugeWidgetDemo")

## Usage:
- Multiple vísualisatins
- Show speed
- fuel indicator
- level indicator
- many more

## How to impletemnt analogWidget to your QMainWindow
1. Open your QtGUI MainWindow with QtDesigner
2. add QWidget from Palette
3. add custom widget to QWidget
![AnalogGaugeWidgetDemo Image](img/3._Add_custom_widget.JPG?raw=true "Add custom widget")

## Update Value Method
update_value(int) -> will also redraw the widget

## Value Changed Signal
valueChanged(int)

## Parameters
- Just play with the demo
![AnalogGaugeWidgetDemo Image](img/Example_without_needle.JPG?raw=true "Add custom widget")
