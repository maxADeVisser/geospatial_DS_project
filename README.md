# geospatial_DS_project
A repo for GeoSpatial Data Science exam project

# Data
The data can be found [here](https://web.ais.dk/aisdata/)

## AIS data attributes
| Columns in *.csv file | Format |
| --- | --- |
| Timestamp | Timestamp from the AIS basestation, format: 31/12/2015 23:59:59 |
| Type of mobile | Describes what type of target this message is received from (class A AIS Vessel, Class B AIS vessel, etc) |
| MMSI | MMSI number of vessel |
| Latitude | Latitude of message report (e.g. 57,8794) |
| Longitude | Longitude of message report (e.g. 17,9125) |
| Navigational status | Navigational status from AIS message if available, e.g.: 'Engaged in fishing', 'Under way using engine', mv. |
| ROT | Rot of turn from AIS message if available |
| SOG | Speed over ground from AIS message if available |
| COG | Course over ground from AIS message if available |
| Heading | Heading from AIS message if available |
| IMO | IMO number of the vessel |
| Callsign | Callsign of the vessel |
| Name | Name of the vessel |
| Ship type | Describes the AIS ship type of this vessel |
| Cargo type | Type of cargo from the AIS message |
| Width | Width of the vessel |
| Length | Lenght of the vessel |
| Type of position fixing device | Type of positional fixing device from the AIS message |
| Draught | Draugth field from AIS message |
| Destination | Destination from AIS message |
| ETA | Estimated Time of Arrival, if available |
| Data source type | Data source type, e.g. AIS |
| Size A | Length from GPS to the bow |
| Size B | Length from GPS to the stern |
| Size C | Length from GPS to starboard side |
| Size D | Length from GPS to port side |
