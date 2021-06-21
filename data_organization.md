# Data Organization

There is a separate subdirectory for every ZInD home, which is named by a unique, sequential ID (e.g., "000", "001", ...).
Within each scene directory, there are separate directories for different types of data as follows:

```
<scene_id>
├── panos
│   └── floor_<floor_id>_partial_room_<partial_room_id>_pano_<pano_id>.jpg
├── floor_plans
│   └── floor_<floor_id>.png
└── zind_data.json
```

# Annotation Format

We provide the **raw** (human-annotated) layouts for each panorama as well as the **derived** **complete** and **visible** layouts from each panorama's point of view.

**ZInD annotations (`zind_data.json`)**:

```
zind_data.json
├── scale_meters_per_coordinate
│   └── floor_<floor_id>                         : float
│
├── merger
│   └── floor_<floor_id>
│       └── complete_room_<complete_room_id>
│           └── partial_room_<partial_room_id>
│               └── pano_<pano_id>
│                   └── label                    : str
│                   └── is_primary               : bool
│                   └── is_inside                : bool
│                   └── is_ceiling_flat          : bool
│                   └── ceiling_height           : float
│                   └── camera_height            : float
│                   └── floor_number             : int
│                   └── image_path               : str
│                   └── layout_raw               : layout
│                   └── layout_complete          : layout
│                   └── layout_visible           : layout
│
├── redraw
│   └── floor_<floor_id>
│       └── room_<room_id>
│           └── vertices                         : List[Tuple[float, float]]
│           └── windows                          : List[Tuple[float, float]]
│           └── doors                            : List[Tuple[float, float]]
│           └── pins
│               └── <pin_id>
│                   └── position                 : Tuple[float, float]
│                   └── label                    : str
│
├── floorplan_to_redraw_transformation
│   └── floor_<floor_id>
│       └── translation                         : Tuple[float, float]
│       └── rotation                            : float
│       └── scale                               : float
│       └── image_path                          : str
└──
```

**Layout annotations**: the room layout alongside 2D bounding boxes for windows, doors and openings in local coordinates.

```
layout
├── vertices                                    : List[Tuple[float, float]]
├── windows                                     : List[Tuple[float, float]]
├── doors                                       : List[Tuple[float, float]]
├── openings                                    : List[Tuple[float, float]]
├── internal                                    : List[Tuple[float, float]]
└──
```

# Glossary of Terms

| Field Name | Description |
| ---------- | ----------- |
| camera_height | Subfield within the panorama field containing the height of the camera used to capture the panorama. It is normalized to 1.0. Note that the normalization applies to all other geometric structures such as ceiling heights, windows, and doors. |
| ceiling_height | Subfield within the panorama field containing the height of the ceiling. |
| complete_room_id | Subfield within the floor field containing information on a complete room, which includes multiple partial room information. |
| doors | Subfield containing information on door location (found under a few different fields). |
| floor_id | Subfield within "merger" containing information on a floor. |
| floor_number | Subfield within the panorama field containing the floor index number. |
| floorplan_to_redraw_transformation | Top level field containing the transformation that maps the identified floor plan(s) to the redraw geometry(ies). |
| floor_plan_transformation | Subfield within the panorama field containing geometric transformation of the room geometry relative to the global floor plan coordinate system. |
| image_path | Subfield within the panorama field containing the relative directory path where the panorama can be found. |
| internal | Subfield containing vertex information on polygon that is not connected to the room layout. Example: polygon of kitchen island that is not connected to the wall. |
| is_ceiling_flat | Subfield within the panorama field containing a boolean flag to indicate if the ceiling for the room is flat (if not, it is "complex"). |
| is_inside | Subfield within the panorama field containing a boolean flag to indicate if the panorama is taken inside or outside the annotated layout. |
| is_primary | Subfield within the panorama field containing a boolean flag to indicate if the panorama is primary (used by annotator to generate room geometry) or not (i.e., panorama is secondary). |
| label | Subfield within the panorama field containing the label of the room, e.g., "kitchen". |
| layout_complete | Subfield within the panorama field containing the geometry after the post-processing has been applied to combine partial room geometries. |
| layout_raw | Subfield within the panorama field containing geometry as originally annotated. |
| layout_visible | Subfield within the panorama field containing all geometries visible from the current panorama’s perspective. |
| merger | Top-level structure containing floor and room information, including panorama coordinates and locations of windows, doors, and openings. |
| openings | Subfield containing information on opening location. |
| pano_id | Subfield within the partial room field containing panorama-related geometry information. |
| partial_room_id | Subfield within the complete room field containing information on a partial room, include information on the panorama used to generate its shape, locations of windows, doors, and openings, and other fields. |
| pins | Subfield within "redraw" rooms containing semantic regional designations. |
| redraw | Top level-structure containing "cleaned up" geometry information (as the final diagram shown to users). In addition to room geometry, windows, and doors, it also contains “pins” as labels. |
| scale_meters_per_coordinate | Scale factor that maps coordinates of a given floor to actual scale in meters. |
| vertices | Subfield containing list of 2D vertices (found under a few different fields). |
| windows | Subfield containing information on window location (found under a few different fields). |




