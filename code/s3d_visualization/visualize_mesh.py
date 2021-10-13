# #!/usr/bin/env python3
# """
# Script to visualize data in wireframe and plane styles

# Example usage:
# python visualize_mesh.py --path /path/to/dataset --scene scene_id

import os
import json
import argparse

import cv2
import open3d
import numpy as np
from panda3d.core import Triangulator

from visualize_3d import convert_lines_to_vertices, clip_polygon

def xyz_2_coorxy(xs, ys, zs, H=512, W=1024):
    us = np.arctan2(xs, ys)
    vs = -np.arctan(zs / np.sqrt(xs**2 + ys**2))
    coorx = (us / (2 * np.pi) + 0.5) * W
    coory = (vs / np.pi + 0.5) * H
    return coorx, coory

def E2P(image, corner_i, corner_j, wall_height, camera, resolution=512, is_wall=True, is_partial=False):
    """convert panorama to persepctive image
    """
    corner_i = corner_i - camera
    corner_j = corner_j - camera

    if is_wall:
        xs = np.linspace(corner_i[0], corner_j[0], resolution)[None].repeat(resolution, 0)
        ys = np.linspace(corner_i[1], corner_j[1], resolution)[None].repeat(resolution, 0)
        if is_partial:
            zs = np.linspace(corner_i[2], corner_j[2], resolution)[:, None].repeat(resolution, 1)
        else:
            zs = np.linspace(-camera[-1], wall_height - camera[-1], resolution)[:, None].repeat(resolution, 1)
    else:
        xs = np.linspace(corner_i[0], corner_j[0], resolution)[None].repeat(resolution, 0)
        ys = np.linspace(corner_i[1], corner_j[1], resolution)[:, None].repeat(resolution, 1)
        zs = np.zeros_like(xs) + wall_height - camera[-1]

    coorx, coory = xyz_2_coorxy(xs, ys, zs, H=image.shape[0], W=image.shape[1])

    persp = cv2.remap(image, coorx.astype(np.float32), coory.astype(np.float32),
                      cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

    return persp


def create_plane_mesh(vertices_walls, vertices_floor, textures, texture_floor, texture_ceiling,
    delta_height, ignore_ceiling=False, walls_tessellation=None, walls_normal=None):
    # create mesh for 3D floorplan visualization
    triangles = []
    triangle_uvs = []

    # the number of vertical walls
    num_walls = len(vertices_walls)

    # 1. vertical wall (always rectangle)
    num_vertices = 0
    if walls_tessellation is None:
        vertices = vertices_walls
    else:
        vertices = []
    floor_centroid = np.mean(vertices_floor, axis=0)
    normals_list = []
    for i in range(len(vertices_walls)):
        # hardcode triangles for each vertical wall
        if walls_tessellation is None:
            triangle = np.array([[0, 2, 1], [2, 0, 3]])
            triangles.append(triangle + num_vertices)
            num_vertices += 4

            triangle_uv = np.array(
                [
                    [i / (num_walls + 2), 0],
                    [i / (num_walls + 2), 1],
                    [(i + 1) / (num_walls + 2), 1],
                    [(i + 1) / (num_walls + 2), 0],
                ],
                dtype=np.float32,
            )
            triangle_uvs.append(triangle_uv)
        else:
            wall_vertices = walls_tessellation[i][0]
            wall_faces = walls_tessellation[i][1]

            wall_axes_y = vertices_walls[i][1] - vertices_walls[i][0]
            wall_axes_y /= np.linalg.norm(wall_axes_y)

            wall_axes_x = vertices_walls[i][3] - vertices_walls[i][0]
            wall_axes_x /= np.linalg.norm(wall_axes_x)

            wall_height = np.linalg.norm(vertices_walls[i][1] - vertices_walls[i][0])
            wall_width = np.linalg.norm(vertices_walls[i][3] - vertices_walls[i][0])

            for vert in wall_vertices:
                vertices.append(vert)

                vert_diff = vert - vertices_walls[i][0]
                texture_dist = 1.0 / (num_walls + 2.0)

                texture_u = (
                    np.clip(np.dot(vert_diff, wall_axes_x) / wall_width, 0, 1) + i
                ) * texture_dist
                texture_v = np.clip(np.dot(vert_diff, wall_axes_y) / wall_height, 0, 1)
                triangle_uvs.append([texture_u, texture_v])

            for face in wall_faces:
                vert0 = np.asarray(wall_vertices[face[0]])
                vert1 = np.asarray(wall_vertices[face[1]])
                vert2 = np.asarray(wall_vertices[face[2]])
                vert_centroid = (vert0 + vert1 + vert2) / 3.0

                # We make sure all walls are pointing towards the interior
                face_normal = np.cross(vert1 - vert0, vert2 - vert1)
                face_floor_dot = np.dot(face_normal, floor_centroid - vert_centroid)
                if face_floor_dot < 0:
                    face = np.flip(face)

                # Make sure the face normals are pointing inside
                triangles.append(face + num_vertices)
                normals_list.append(walls_normal[i])

            num_vertices += len(wall_vertices)

    if walls_tessellation is not None:
        vertices = [np.asarray(vertices)]
        triangles = [np.asarray(triangles)]
        triangle_uvs = [np.asarray(triangle_uvs)]

    # 2. floor and ceiling
    # Since the floor and ceiling may not be a rectangle, triangulate the polygon first.
    tri = Triangulator()
    for i in range(len(vertices_floor)):
        tri.add_vertex(vertices_floor[i, 0], vertices_floor[i, 1])

    for i in range(len(vertices_floor)):
        tri.add_polygon_vertex(i)

    tri.triangulate()

    # polygon triangulation
    triangle = []
    for i in range(tri.getNumTriangles()):
        triangle.append([tri.get_triangle_v0(i), tri.get_triangle_v1(i), tri.get_triangle_v2(i)])
    triangle = np.array(triangle)

    # add triangles for floor and ceiling
    triangles.append(triangle + num_vertices)
    num_vertices += len(np.unique(triangle))
    if not ignore_ceiling:
        triangles.append(triangle + num_vertices)

    # texture for floor and ceiling
    vertices_floor_min = np.min(vertices_floor[:, :2], axis=0)
    vertices_floor_max = np.max(vertices_floor[:, :2], axis=0)

    # normalize to [0, 1]
    triangle_uv = (vertices_floor[:, :2] - vertices_floor_min) / (vertices_floor_max - vertices_floor_min)
    triangle_uv[:, 0] = (triangle_uv[:, 0] + num_walls) / (num_walls + 2)

    triangle_uvs.append(triangle_uv)

    # normalize to [0, 1]
    triangle_uv = (vertices_floor[:, :2] - vertices_floor_min) / (vertices_floor_max - vertices_floor_min)
    triangle_uv[:, 0] = (triangle_uv[:, 0] + num_walls + 1) / (num_walls + 2)

    triangle_uvs.append(triangle_uv)

    # 3. Merge wall, floor, and ceiling
    vertices.append(vertices_floor)
    vertices.append(vertices_floor + delta_height)
    vertices = np.concatenate(vertices, axis=0)

    triangles = np.concatenate(triangles, axis=0)

    textures.append(texture_floor)
    textures.append(texture_ceiling)
    textures = np.concatenate(textures, axis=1)

    triangle_uvs = np.concatenate(triangle_uvs, axis=0)

    mesh = open3d.geometry.TriangleMesh(
        vertices=open3d.utility.Vector3dVector(vertices),
        triangles=open3d.utility.Vector3iVector(triangles)
    )
    mesh.compute_vertex_normals()

    mesh.texture = open3d.geometry.Image(textures)
    mesh.triangle_uvs = np.array(triangle_uvs[triangles.reshape(-1), :], dtype=np.float64)
    return mesh


def verify_normal(corner_i, corner_j, delta_height, plane_normal):
    edge_a = corner_j + delta_height - corner_i
    edge_b = delta_height

    normal = np.cross(edge_a, edge_b)
    normal /= np.linalg.norm(normal, ord=2)

    inner_product = normal.dot(plane_normal)

    if inner_product > 1e-8:
        return False
    else:
        return True


def visualize_mesh(args):
    """visualize as water-tight mesh
    """

    print(os.path.join(args.path, "scene_{:05d}".format(int(args.scene)), "2D_rendering", str(args.room),
            "panorama/full/rgb_rawlight.png"))

    image = cv2.imread(os.path.join(args.path, "scene_{:05d}".format(int(args.scene)), "2D_rendering",
            str(args.room), "panorama/full/rgb_rawlight.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # load room annotations
    with open(os.path.join(args.path, "scene_{:05d}".format(int(args.scene)), "annotation_3d.json")) as f:
        annos = json.load(f)

    # load camera info
    camera_center = np.loadtxt(os.path.join(args.path, "scene_{:05d}".format(int(args.scene)), "2D_rendering",
                                            str(args.room), "panorama", "camera_xyz.txt"))

    # parse corners
    junctions = np.array([item['coordinate'] for item in annos['junctions']])
    lines_holes = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist())

    lines_holes = np.unique(lines_holes)
    _, vertices_holes = np.where(np.array(annos['lineJunctionMatrix'])[lines_holes])
    vertices_holes = np.unique(vertices_holes)

    # parse annotations
    walls = dict()
    walls_normal = dict()
    walls_split_dict = dict()
    for semantic in annos['semantics']:
        if semantic['ID'] != int(args.room):
            continue

        # find junctions of ceiling and floor
        for planeID in semantic['planeID']:
            plane_anno = annos['planes'][planeID]

            if plane_anno['type'] != 'wall':
                lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0]
                lineIDs = np.setdiff1d(lineIDs, lines_holes)
                junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
                wall = convert_lines_to_vertices(junction_pairs)
                if len(wall) > 0:
                    walls[plane_anno['type']] = wall[0]

        # save normal of the vertical walls
        for planeID in semantic['planeID']:
            plane_anno = annos['planes'][planeID]

            if plane_anno['type'] == 'wall':
                plane_anno = annos['planes'][planeID]
                lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
                junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
                wall = convert_lines_to_vertices(junction_pairs)
                vertices, faces = clip_polygon(wall, vertices_holes, junctions, plane_anno)
                wall_wdo_intersection = tuple(np.intersect1d(wall, walls["floor"]))

                walls_normal[wall_wdo_intersection] = plane_anno["normal"]
                walls_split_dict[wall_wdo_intersection] = (vertices, faces)

    # we assume that zs of floor equals 0, then the wall height is from the ceiling
    if len(walls) > 0:
        wall_height = np.mean(junctions[walls['ceiling']], axis=0)[-1]
        delta_height = np.array([0, 0, wall_height])

        # list of corner index
        wall_floor = walls['floor']

        corners = []  # 3D coordinate for each wall
        textures = []  # texture for each wall
        walls_tessellation = ([])  # Tessellation for each wall, potentially with holes from WDO elements.
        normals_list = []

        # wall
        for i, j in zip(wall_floor, np.roll(wall_floor, shift=-1)):
            corner_i, corner_j = junctions[i], junctions[j]

            wall_id = tuple(sorted([i, j]))
            wall_split_tuple = walls_split_dict[wall_id]
            wall_normal = walls_normal[tuple(sorted([i, j]))]
            flip = verify_normal(corner_i, corner_j, delta_height, wall_normal)

            if flip:
                corner_j, corner_i = corner_i, corner_j

            texture = E2P(image, corner_i, corner_j, wall_height, camera_center)

            corner = np.array(
                [corner_i, corner_i + delta_height, corner_j + delta_height, corner_j]
            )

            corners.append(corner)
            textures.append(texture)
            walls_tessellation.append(wall_split_tuple)
            normals_list.append(wall_normal)

        # floor and ceiling
        # the floor/ceiling texture is cropped by the maximum bounding box
        corner_floor = junctions[wall_floor]
        corner_min = np.min(corner_floor, axis=0)
        corner_max = np.max(corner_floor, axis=0)
        texture_floor = E2P(
            image, corner_min, corner_max, 0, camera_center, is_wall=False
        )
        texture_ceiling = E2P(
            image, corner_min, corner_max, wall_height, camera_center, is_wall=False
        )

        # create mesh
        mesh = create_plane_mesh(corners, corner_floor, textures, texture_floor, texture_ceiling,
            delta_height, ignore_ceiling=args.ignore_ceiling, walls_tessellation=walls_tessellation,walls_normal=normals_list)
    else:
        mesh = None

    camera_mesh = open3d.geometry.TriangleMesh.create_sphere(0.05)
    camera_mesh.compute_vertex_normals()
    camera_mesh.translate(camera_center)

    if mesh is None:
        camera_mesh.paint_uniform_color([0.0, 1.0, 0.0])
    else:
        camera_mesh.paint_uniform_color([1.0, 0.0, 0.0])

    return mesh, camera_mesh


def visualize_full_mesh(args):
    """
    visualize perspective layout
    """

    scene_path = os.path.join(
        args.path, "scene_{:05d}".format(int(args.scene)), "2D_rendering"
    )
    mesh_list = []
    for room_index, room_id in enumerate(np.sort(os.listdir(scene_path))):
        room_folder = os.path.join(scene_path, room_id)
        pano_file = os.path.join(room_folder, "panorama", "full", "rgb_rawlight.png")

        if os.path.isfile(pano_file):
            args.room = room_id
            try:
                mesh, camera_mesh = visualize_mesh(args)
                if mesh is not None:
                    mesh_list.append(mesh)
                mesh_list.append(camera_mesh)
            except Exception as ex:
                print("Error rendering {}: {}".format(room_id, str(ex)))

    def fix_bug(v):
        v.update_geometry(None)
        return False

    open3d.visualization.draw_geometries_with_animation_callback(mesh_list, fix_bug)


def parse_args():
    parser = argparse.ArgumentParser(description="Structured3D 2D Layout Visualization")
    parser.add_argument("--path", required=True,
                        help="dataset path", metavar="DIR")
    parser.add_argument("--scene", required=True,
                        help="scene id", type=int)
    parser.add_argument("--room", required=False,
                        help="room id", type=int)
    parser.add_argument("--ignore_ceiling", action='store_true',
                        help="ignore ceiling for better visualization")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.room is None:
        visualize_full_mesh(args)
    else:
        visualize_mesh(args)


if __name__ == "__main__":
    main()
