# #!/usr/bin/env python3
# """
# Script to visualize data in wireframe and plane styles

# Example usage:
# python visualize_3d.py --path /path/to/dataset --scene scene_id --type wireframe/plane

import os
import json
import argparse

import open3d
import pymesh
import numpy as np

# define color map
colormap_255 = [
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 190],
    [0, 128, 128],
    [230, 190, 255],
    [170, 110, 40],
    [255, 250, 200],
    [128, 0, 0],
    [170, 255, 195],
    [128, 128, 0],
    [255, 215, 180],
    [0, 0, 128],
    [128, 128, 128],
    [255, 255, 255],
    [0, 0, 0],
]


def visualize_wireframe(annos):
    """visualize wireframe
    """
    colormap = np.array(colormap_255) / 255

    junctions = np.array([item['coordinate'] for item in annos['junctions']])
    _, junction_pairs = np.where(np.array(annos['lineJunctionMatrix']))
    junction_pairs = junction_pairs.reshape(-1, 2)

    # extract hole lines
    lines_holes = []
    lines_windows = []
    lines_doors = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_curr = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
                lines_holes.extend(lines_curr)
                if semantic['type'] in ['window']:
                    lines_windows.extend(lines_curr)
                else:
                    lines_doors.extend(lines_curr)

    lines_holes = np.unique(lines_holes)

    # extract cuboid lines
    cuboid_lines = []
    for cuboid in annos['cuboids']:
        for planeID in cuboid['planeID']:
            cuboid_lineID = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
            cuboid_lines.extend(cuboid_lineID)
    cuboid_lines = np.unique(cuboid_lines)
    cuboid_lines = np.setdiff1d(cuboid_lines, lines_holes)

    # visualize junctions
    connected_junctions = junctions[np.unique(junction_pairs)]
    connected_colors = np.repeat(colormap[0].reshape(1, 3), len(connected_junctions), axis=0)

    junction_set = open3d.geometry.PointCloud()
    junction_set.points = open3d.utility.Vector3dVector(connected_junctions)
    junction_set.colors = open3d.utility.Vector3dVector(connected_colors)

    # visualize line segments
    line_colors = np.repeat(colormap[5].reshape(1, 3), len(junction_pairs), axis=0)

    # color holes
    if len(lines_holes) != 0:
        line_colors[lines_holes] = colormap[6]
    # color windows
    if len(lines_windows) != 0:
        line_colors[lines_windows] = colormap[6]
    # color doors
    if len(lines_doors) != 0:
        line_colors[lines_doors] = colormap[-6]
    # color cuboids
    if len(cuboid_lines) != 0:
        line_colors[cuboid_lines] = colormap[2]

    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(junctions)
    line_set.lines = open3d.utility.Vector2iVector(junction_pairs)
    line_set.colors = open3d.utility.Vector3dVector(line_colors)

    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Info)
    open3d.visualization.draw_geometries([junction_set, line_set])


def project(x, meta):
    """ project 3D to 2D for polygon clipping
    """
    proj_axis = max(range(3), key=lambda i: abs(meta['normal'][i]))

    return tuple(c for i, c in enumerate(x) if i != proj_axis)


def project_inv(x, meta):
    """ recover 3D points from 2D
    """
    # Returns the vector w in the walls' plane such that project(w) equals x.
    proj_axis = max(range(3), key=lambda i: abs(meta['normal'][i]))

    w = list(x)
    w[proj_axis:proj_axis] = [0.0]
    c = -meta['offset']
    for i in range(3):
        c -= w[i] * meta['normal'][i]
    c /= meta['normal'][proj_axis]
    w[proj_axis] = c
    return tuple(w)


def triangulate(points):
    """ triangulate the plane for operation and visualization
    """

    num_points = len(points)
    indices = np.arange(num_points, dtype=np.int)
    segments = np.vstack((indices, np.roll(indices, -1))).T

    tri = pymesh.triangle()
    tri.points = np.array(points)

    tri.segments = segments
    tri.verbosity = 0
    tri.run()

    return tri.mesh


def clip_polygon(polygons, vertices_hole, junctions, meta):
    """ clip polygon the hole
    """
    if len(polygons) == 1:
        junctions = [junctions[vertex] for vertex in polygons[0]]
        mesh_wall = triangulate(junctions)

        vertices = np.array(mesh_wall.vertices)
        faces = np.array(mesh_wall.faces)

        return vertices, faces

    else:
        wall = []
        holes = []
        for polygon in polygons:
            if np.any(np.intersect1d(polygon, vertices_hole)):
                holes.append(polygon)
            else:
                wall.append(polygon)

        # extract junctions on this plane
        indices = []
        junctions_wall = []
        for plane in wall:
            for vertex in plane:
                indices.append(vertex)
                junctions_wall.append(junctions[vertex])

        junctions_holes = []
        for plane in holes:
            junctions_hole = []
            for vertex in plane:
                indices.append(vertex)
                junctions_hole.append(junctions[vertex])
            junctions_holes.append(junctions_hole)

        junctions_wall = [project(x, meta) for x in junctions_wall]
        junctions_holes = [[project(x, meta) for x in junctions_hole] for junctions_hole in junctions_holes]

        mesh_wall = triangulate(junctions_wall)

        for hole in junctions_holes:
            mesh_hole = triangulate(hole)
            mesh_wall = pymesh.boolean(mesh_wall, mesh_hole, 'difference')

        vertices = [project_inv(vertex, meta) for vertex in mesh_wall.vertices]

        return vertices, np.array(mesh_wall.faces)


def draw_geometries_with_back_face(geometries):
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window()


def convert_lines_to_vertices(lines):
    """convert line representation to polygon vertices
    """
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons


def visualize_plane(annos, args, eps=0.9):
    """visualize plane
    """
    colormap = np.array(colormap_255) / 255
    junctions = [item['coordinate'] for item in annos['junctions']]

    if args.color == 'manhattan':
        manhattan = dict()
        for planes in annos['manhattan']:
            for planeID in planes['planeID']:
                manhattan[planeID] = planes['ID']

    # extract hole vertices
    lines_holes = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist())

    lines_holes = np.unique(lines_holes)
    if len(lines_holes) > 0:
        _, vertices_holes = np.where(np.array(annos['lineJunctionMatrix'])[lines_holes])
        vertices_holes = np.unique(vertices_holes)
    else:
        vertices_holes = []

    # load polygons
    polygons = []
    for semantic in annos['semantics']:
        # Window/door/opening is not rendered so that they can show as holes
        if semantic['type'] in ['door', 'window']:
            continue
        for planeID in semantic['planeID']:
            plane_anno = annos['planes'][planeID]
            lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
            junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
            polygon = convert_lines_to_vertices(junction_pairs)
            vertices, faces = clip_polygon(polygon, vertices_holes, junctions, plane_anno)
            polygons.append([vertices, faces, planeID, plane_anno['normal'], plane_anno['type'], semantic['type']])

    plane_set = []
    for i, (vertices, faces, planeID, normal, plane_type, semantic_type) in enumerate(polygons):
        # ignore the room ceiling
        if plane_type == 'ceiling' and semantic_type not in ['door', 'window']:
            continue

        plane_vis = open3d.geometry.TriangleMesh()

        plane_vis.vertices = open3d.utility.Vector3dVector(vertices)
        plane_vis.triangles = open3d.utility.Vector3iVector(faces)

        if args.color == 'normal':
            if np.dot(normal, [1, 0, 0]) > eps:
                plane_vis.paint_uniform_color(colormap[0])
            elif np.dot(normal, [-1, 0, 0]) > eps:
                plane_vis.paint_uniform_color(colormap[1])
            elif np.dot(normal, [0, 1, 0]) > eps:
                plane_vis.paint_uniform_color(colormap[2])
            elif np.dot(normal, [0, -1, 0]) > eps:
                plane_vis.paint_uniform_color(colormap[3])
            elif np.dot(normal, [0, 0, 1]) > eps:
                plane_vis.paint_uniform_color(colormap[4])
            elif np.dot(normal, [0, 0, -1]) > eps:
                plane_vis.paint_uniform_color(colormap[5])
            else:
                plane_vis.paint_uniform_color(colormap[6])
        elif args.color == 'manhattan':
            # paint each plane with manhattan world
            if planeID not in manhattan.keys():
                plane_vis.paint_uniform_color(colormap[6])
            else:
                plane_vis.paint_uniform_color(colormap[manhattan[planeID]])

        plane_set.append(plane_vis)

    draw_geometries_with_back_face(plane_set)


def parse_args():
    parser = argparse.ArgumentParser(description="Structured3D 3D Visualization")
    parser.add_argument("--path", required=True,
                        help="dataset path", metavar="DIR")
    parser.add_argument("--scene", required=True,
                        help="scene id", type=int)
    parser.add_argument("--type", choices=("wireframe", "plane"),
                        default="plane", type=str)
    parser.add_argument("--color", choices=["normal", "manhattan"],
                        default="normal", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    # load annotations from json
    with open(os.path.join(args.path, f"scene_{args.scene:05d}", "annotation_3d.json")) as file:
        annos = json.load(file)

    if args.type == "wireframe":
        visualize_wireframe(annos)
    elif args.type == "plane":
        visualize_plane(annos, args)


if __name__ == "__main__":
    main()
