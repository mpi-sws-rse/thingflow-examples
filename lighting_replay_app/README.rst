===========================
Lighting Replay Application
===========================

This directory contains an end-to-end application which monitors the
light levels in several rooms of a residence and then replays a similar
pattern when it is unoccupied. This demonstrates the use of ThingFlow
in a more complex scenario.

The file `baypiggies-iot-talk-oct-2016.pdf <https://github.com/mpi-sws-rse/thingflow-examples/blob/master/lighting_replay_app/baypiggies-iot-talk-oct-2016-revised.pdf>`__
contains the slides for a talk
describing this application: *Python and IoT: From Chips and Bits to
Data Science*. The talk was given at the Bay Area
Python Interest Group on October 27, 2016. It was updated in April 2017
to reflect the name change of AntEvents to ThingFlow.

Data Capture
------------
The ``capture`` subdirectory contains the code to capture and record light
sensor readings.

Light sensor data is gathered through one or more ESP8266 boards running
Micropython and sent over an MQTT queue to a central server app running
on the Raspberry Pi. The Raspberry Pi reads from the queue and saves the
data to flat csv files. It also has a light sensor, which it samples and
saves to a flat file as well.

Examples of captured data from three different rooms may be found in the
``data_files`` subdirectory.

Data Analysis
-------------
The light sensor data is next analyzed. The ``analysis`` subdirectory
contains Jupyter notebooks (.ipynb files) analyzing the data in
``data_files``. The csv files are parsed, post-processed, and read into
`Pandas <http://pandas.pydata.org/>`__ ``Series`` data structures.

From there, the light readings are grouped into four levels via
K-Means clustering. The four levels are then mapped to on-off values,
depending on the particulars of each room (e.g. how much ambiant light
is present). We divide each day into four "zones", based on the absolute
time of day and sunrise/sunset times. The samples are grouped into
subsequences separating them by zone and when there is a gap in the
data readings.

These subsequences are then used to train Hidden Markov Models
using `hmmlearn <https://github.com/hmmlearn/hmmlearn>`__. Hmmlearn
can infer a state machine which can emit a similar pattern of on/off samples.
A total of four models are created per room, with one for each zone.

Player
------
The ``player`` subdirectory contains a light controller application which
runs off the Hidden Markov Models created in the analysis phase. It controls
`Philips Hue <http://www.developers.meethue.com/>`__ smart
lights, which are accessible via an REST api. We use the
`phue <https://pypi.python.org/pypi/phue/0.8>`__ library to abstract the
details of the control protocol.

