#include <map>
#include <tuple>
#ifndef CITY
#define CITY

#include <cinolib/meshes/meshes.h>
#include <cinolib/merge_meshes_at_coincident_vertices.h>
#include <cinolib/connected_components.h>
// #include <cinolib/drawable_segment_soup.h>

#include <ground.h>
#include <building.h>
#include <street.h>
#include <auxiliary.h>

namespace cinolib
{

using std::vector;
using std::string;

class City
{
private:
    Polygonmesh<>    boundary;
    Trimesh<>        city_mesh;
    Trimesh<>        buildings_mesh;
    Ground           ground;
    vector<Building> buildings;
    vector<Street>   streets;
    bool             WITH_STREETS;
    vec3d            scene_center = vec3d(0,0,0);

    void mark_streets(const Trimesh<> &m);
    void print_poly_labels(const string &file_path) const;
    void print_edge_labels(const string &file_path) const;
    void translate_back();

public:
    City() : ground() {}
    ~City() {}

    void load_boundary_polygon  (const string &filepath);
    void load_buildings_data    (const string &dir_path);
    void load_streets_data      (const string &filepath);

    void compute_ground_mesh();
    void compute_buildings_mesh();
    void merge_meshes();

    void verify_mesh(const Trimesh<> &m);
    void save(const string &dir_path, bool save_parts = false);
    void write_streets_data(const string &filepath);
    // void visualize_streets(DrawableSegmentSoup &soup) const;

    uint n_buildings() const { return buildings.size(); }
    uint n_streets()   const { return streets.size();   }
    vec3d get_scene_center() const { return scene_center; }

    Trimesh<> get_city_mesh()       const { return city_mesh; }
    const Trimesh<> &get_city_mesh_ref() const { return city_mesh; }
    const Trimesh<> *get_city_mesh_ptr() const { return &city_mesh; }

    Trimesh<> get_buildings_mesh()       const { return buildings_mesh; }
    const Trimesh<> &get_buildings_mesh_ref() const { return buildings_mesh; }
    const Trimesh<> *get_buildings_mesh_ptr() const { return &buildings_mesh; }

    Trimesh<> get_ground_mesh()       const { return ground.get_ground(); }
    const Trimesh<> &get_ground_mesh_ref() const { return ground.get_ground_ref(); }
    const Trimesh<> *get_ground_mesh_ptr() const { return ground.get_ground_ptr(); }

    vector<Building> get_buildings() const { return buildings; }
    const vector<Building> &get_buildings_ref() const { return buildings; }

    vector<Street>   get_streets()   const { return streets;   }
    const vector<Street>   &get_streets_ref()   const { return streets;   }
    void set_streets(const vector<Street> &S) { streets = S; }
};

// --------------------------------------------------------------------------------------------

void City::translate_back()
{
    vec3d c = scene_center /*- city_mesh.centroid()*/;
    city_mesh.translate(c);
    buildings_mesh.translate(c);
    boundary.translate(c);
}

// --------------------------------------------------------------------------------------------

void City::compute_ground_mesh()
{
    ground.setup(boundary, buildings, streets, scene_center);
    ground.compute();
    // ground.verify_mesh();
}

// --------------------------------------------------------------------------------------------

void City::compute_buildings_mesh()
{
    buildings_mesh.clear();
    int count = 0;
    for (Building &B : buildings) {
        std::cout << "\r\033[K"; // deletes the line
        std::cout << "City::compute_buildings_mesh - building " << count << " / "
                  << buildings.size() << std::flush;
        if (!B.COMPUTED) {
            auto buf = suppress_stdout();
            B.create_building_mesh();
            restore_stdout(buf);
        }
        // Append the building directly.  Buildings do not share vertices in
        // normal footprints, so rebuilding an octree for every append (as the
        // previous merge did) only made this step quadratic.  Shared base
        // vertices are snapped against the ground in merge_meshes.
        const Polygonmesh<> &bm = B.get_building_ref();
        uint v_offset = buildings_mesh.num_verts();
        for (uint vid=0; vid<bm.num_verts(); ++vid) {
            buildings_mesh.vert_add(bm.vert(vid));
        }
        int building_ID = B.get_building_ID();
        for (uint pid=0; pid<bm.num_polys(); ++pid) {
            std::vector<uint> p = bm.poly_verts_id(pid);
            for (uint &vid : p) vid += v_offset;
            uint fresh_id = buildings_mesh.poly_add(p);
            buildings_mesh.poly_data(fresh_id).label = building_ID;
        }
        ++count;
    }
    buildings_mesh.update_bbox();
    std::cout << std::endl;
    // verify_mesh(buildings_mesh);
}

// --------------------------------------------------------------------------------------------

void City::merge_meshes()
{
    const Trimesh<> *ground_mesh = ground.get_ground_ptr();

    city_mesh.clear();

    // Append the ground first and remember every vertex by its XY position.
    Vec2KeyMap vertex_map;
    vertex_map.reserve(ground_mesh->num_verts() * 2 + buildings_mesh.num_verts());
    for (uint vid=0; vid<ground_mesh->num_verts(); ++vid) {
        vec3d p = ground_mesh->vert(vid);
        uint fresh_id = city_mesh.vert_add(p);
        vertex_map.emplace(Vec2Key{p.x(), p.y()}, fresh_id);
    }
    for (uint pid=0; pid<ground_mesh->num_polys(); ++pid) {
        std::vector<uint> p = ground_mesh->poly_verts_id(pid);
        uint fresh_id = city_mesh.poly_add(p);
        city_mesh.poly_data(fresh_id).label = ground_mesh->poly_data(pid).label;
    }

    // Snap building vertices to the ground rim by XY, so walls and ground
    // are welded even when the two height interpolations differ slightly.
    std::map<std::tuple<double, double, double>, uint> building_vertices;
    std::vector<uint> remap(buildings_mesh.num_verts());
    for (uint vid=0; vid<buildings_mesh.num_verts(); ++vid) {
        vec3d p = buildings_mesh.vert(vid);
        Vec2Key key{p.x(), p.y()};
        auto it = vertex_map.find(key);
        if (it != vertex_map.end() &&
            std::abs(city_mesh.vert(it->second).z() - p.z()) < 0.01) {
            remap[vid] = it->second;
        } else {
            auto xyz = std::make_tuple(p.x(), p.y(), p.z());
            auto found = building_vertices.find(xyz);
            if (found != building_vertices.end()) {
                remap[vid] = found->second;
            } else {
                uint fresh_id = city_mesh.vert_add(p);
                building_vertices.emplace(xyz, fresh_id);
                remap[vid] = fresh_id;
            }
        }
    }
    for (uint pid=0; pid<buildings_mesh.num_polys(); ++pid) {
        if (pid%1000 == 0) {
            std::cout << "\r\033[K"; // deletes the line
            std::cout << "City::merge_meshes - poly " << pid << " / "
                      << buildings_mesh.num_polys() << std::flush;
        }
        std::vector<uint> p = buildings_mesh.poly_verts_id(pid);
        for (uint &vid : p) vid = remap[vid];
        uint fresh_id = city_mesh.poly_add(p);
        city_mesh.poly_data(fresh_id).label = buildings_mesh.poly_data(pid).label;
    }
    std::cout << std::endl;
    city_mesh.update_bbox();
    if (WITH_STREETS) {
        mark_streets(*ground_mesh);
    }
    // verify_mesh(city_mesh);
}

// --------------------------------------------------------------------------------------------

void City::verify_mesh(const Trimesh<> &m)
{
    double TOLL_1D = 1e-6;
    double TOLL_2D = 1e-12;

    uint non_manifold_verts = 0;
    for (uint vid=0; vid<m.num_verts(); ++vid) {
        if (!m.vert_is_manifold(vid)) {
            non_manifold_verts++;
            std::cout << "  non-manifold vertex ID: " << vid << " at position " << m.vert(vid) << std::endl;
        }
    }

    uint non_manifold_edges = 0;
    for (uint eid=0; eid<m.num_edges(); ++eid) {
        if (!m.edge_is_manifold(eid)) {
            non_manifold_edges++;
            std::cout << "  non-manifold edge ID: " << eid << " between vertices " << m.edge_vert_ids(eid).front() << " and "
                      << m.edge_vert_ids(eid).back() << std::endl;
        }
    }

    uint small_edges = 0;
    for (uint eid=0; eid<m.num_edges(); ++eid) {
        if (m.edge_length(eid) < TOLL_1D) {
            small_edges++;
        }
    }

    uint small_polys = 0;
    for (uint pid=0; pid<m.num_polys(); ++pid) {
        if (m.poly_area(pid) < TOLL_2D) {
            small_polys++;
        }
    }

    std::vector<vec3d> verts = m.vector_verts();
    REMOVE_DUPLICATES_FROM_VEC(verts);
    uint duplicate_verts = m.num_verts() - verts.size();

    std::vector<std::vector<uint>> polys = m.vector_polys();
    REMOVE_DUPLICATES_FROM_VEC(polys);
    uint duplicate_polys = m.num_polys() - polys.size();

    int euler_char       = m.num_verts() - m.num_edges() + m.num_polys();
    uint conn_components = connected_components(m);
    uint bnd_components  = buildings.size() + 1; // building footprints + boundary
    int holes            = (2 * conn_components - bnd_components - euler_char) / 2;

    std::string message = "Mesh verification: [";
    if (non_manifold_verts > 0)
        message += "\n  non-manifold vertices: " + std::to_string(non_manifold_verts);
    if (non_manifold_edges > 0)
        message += "\n  non-manifold edges: " + std::to_string(non_manifold_edges);
    if (small_edges > 0)
        message += "\n  small edges (< " + std::to_string(TOLL_1D) + "): " + std::to_string(small_edges);
    if (small_polys > 0)
        message += "\n  small polys (< " + std::to_string(TOLL_2D) + "): " + std::to_string(small_polys);
    if (duplicate_verts > 0)
        message += "\n  duplicate vertices: " + std::to_string(duplicate_verts);
    if (duplicate_polys > 0)
        message += "\n  duplicate polys: " + std::to_string(duplicate_polys);
    if (holes != 0)
        message += "\n  holes: " + std::to_string(holes);
    message += "\n]\n";
    std::cout << message << std::flush;
}

// --------------------------------------------------------------------------------------------

void City::mark_streets(const Trimesh<> &m)
{
    if (!WITH_STREETS) return;

    // mark the edges corresponding to streets
    int count = 0;
    for (uint eid=0; eid<m.num_edges(); ++eid) {
        auto vids = m.edge_vert_ids(eid);
        int eid_tri = city_mesh.edge_id(vids);
        if (eid_tri == -1) {
            count++;
            continue;
        }
        city_mesh.edge_data(eid_tri).label = m.edge_data(eid).label;

        // if the edge corresponds to a street, set the z-coordinate of the vertices
        // perchè la z delle strade è sbagliata?
        if (city_mesh.edge_data(eid_tri).label != -1) {
            for (uint vid : city_mesh.edge_vert_ids(eid_tri)) {
                double avg_z = 0.;
                for (uint nbr : city_mesh.adj_v2v(vid)) {
                    avg_z += city_mesh.vert(nbr).z();
                }
                avg_z /= city_mesh.adj_v2v(vid).size();
                city_mesh.vert(vid).z() = avg_z;
            }
        }
    }
    if (count > 0) {
        std::cout << "  City::mark_streets - WARNING: " << count
                  << " edges not found in triangulated ground!" << std::endl;
    }
}

// --------------------------------------------------------------------------------------------

void City::save(const string &dir_path, bool save_parts)
{
    string mesh_filename = dir_path + "/mesh.obj";
    string buildings_filename = dir_path + "/buildings_mesh.obj";
    string ground_filename    = dir_path + "/ground_mesh.obj";

    translate_back();

    city_mesh.poly_color_wrt_label();
    city_mesh.save(mesh_filename.c_str());

    // The merged city mesh is the default product.  The individual parts
    // are only written when explicitly requested.
    if (save_parts) {
        buildings_mesh.poly_color_wrt_label();
        buildings_mesh.save(buildings_filename.c_str());
        ground.save(ground_filename.c_str());
    }
}

// --------------------------------------------------------------------------------------------

// print a csv file with the ID of each poly and its label
void City::print_poly_labels(const string &file_path) const
{
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file.\n";
        return;
    }
    file << "poly_ID building_ID\n";
    for (uint pid=0; pid<city_mesh.num_polys(); ++pid) {
        file << pid << " " << city_mesh.poly_data(pid).label << "\n";
    }
    file.close();
}

// --------------------------------------------------------------------------------------------

// print a csv file with the ID of each poly and its label
void City::print_edge_labels(const string &file_path) const
{
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file.\n";
        return;
    }
    file << "edge_v0_ID edge_v1_ID edge_ID\n";
    for (uint eid=0; eid<city_mesh.num_edges(); ++eid) {
        auto verts = city_mesh.edge_vert_ids(eid);
        file << verts.front() << " " << verts.back() << " " << city_mesh.edge_data(eid).label << "\n";
    }
    file.close();
}

// --------------------------------------------------------------------------------------------

void City::load_boundary_polygon(const string &filepath)
{
    boundary.load(filepath.c_str());
    if (boundary.num_polys() != 1 && !boundary.mesh_is_manifold()) {
        throw std::runtime_error("Error: the boundary mesh should contain a single (manifold) polygon");
    }
    // Use the boundary face only.  The OFF may also carry isolated ground
    // sample points (vertices without faces), which must not bias the center.
    scene_center = vec3d(0,0,0);
    const std::vector<uint> &outer = boundary.adj_p2v(0);
    for (uint vid : outer) scene_center += boundary.vert(vid);
    scene_center /= double(outer.size());
    boundary.translate(-scene_center);
}

// --------------------------------------------------------------------------------------------

void City::load_buildings_data(const string &dir_path)
{
    if (!fs::exists(dir_path)) {
        std::cerr << "load_buildings - ERROR: the buildings path does not exist" << std::endl;
        exit(0);
    }
    for (const auto &entry : fs::directory_iterator(dir_path)) {
        if (!entry.is_directory()) continue;
        string dir            = entry.path().string();
        int    ID             = extract_directory_ID(dir);
        if (ID < 1) continue; // cache/metadata folders are not buildings
        string footprint_file = dir + "/pavement_polygon.off";
        string roof_file      = dir + "/roof_polygon.off";
        string facades_file   = dir + "/facades.off";
        if (!fs::exists(footprint_file) || !fs::exists(roof_file) || !fs::exists(facades_file)) {
            std::cerr << "load_buildings - WARNING: missing footprint, roof, or facades file, "
                      << "discarded Building " << ID << std::endl;
            continue;
        }

        // load data
        // if (ID == 68 ) {
        //     std::cout << "load_buildings - WARNING: could not load footprint mesh, "
        //               << "discarded Building " << ID << std::endl;
        //     continue;
        // }
        auto buf = suppress_stdout();
        Polygonmesh<> m_footprint(footprint_file.c_str());
        Polygonmesh<> m_roof(roof_file.c_str());
        Polygonmesh<> m_facades(facades_file.c_str());
        restore_stdout(buf);
        vec3d building_center = m_footprint.centroid();
        Building_Classification C = Building_Classification::building;

        // check data
        if (!m_footprint.mesh_is_manifold()) {
            std::cout << "load_buildings - WARNING: footprint mesh is not manifold, "
                      << "discarded Building " << ID << std::endl;
            continue;
        }
        if (mesh_has_duplicates(m_footprint)) {
            std::cout << "load_buildings - WARNING: footprint has duplicate vertices, "
                      << "discarded Building " << ID << std::endl;
            continue;
        }

        // check if the roof polygons intersect (I am using the roof here because the footprint may be non planar)
        bool intersect = false;
        for (uint pid0=0; pid0<m_roof.num_polys(); ++pid0) {
            for (uint pid1=1; pid1<m_roof.num_polys(); ++pid1) {
                if (pid0 == pid1) continue;
                if (polygons_intersect(m_roof, pid0, pid1)) {
                    intersect = true;
                    break;
                }
            }
        }
        if (intersect) {
            std::cout << "load_buildings - WARNING: roof contains intersecting polygons, "
                      << "discarded Building " << ID << std::endl;
            continue;
        }

        Building B(m_footprint, m_roof, m_facades, ID, building_center, C);
        B.translate(-scene_center);
        if (!B.is_inside(boundary)) {
            std::cout << "load_buildings - WARNING: footprint is outside the boundary polygon, "
                      << "discarded Building " << ID << std::endl;
            continue;
        }
        buildings.push_back(B);
    }
    if (buildings.empty()) {
        std::cerr << "load_buildings - ERROR: no buildings found!" << std::endl;
        exit(0);
    }
}

// --------------------------------------------------------------------------------------------

void City::load_streets_data(const string &filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cout << "load_streets_GeoJSON - Failed to open Streets file: " << filepath << "\n";
        WITH_STREETS = false;
        return;
    }
    nlohmann::json geojson;
    file >> geojson;
    if (!geojson.contains("features")) {
        std::cerr << "Invalid GeoJSON: no features\n";
    }

    uint count = 0;
    for (const auto &feature : geojson["features"]) {
        if (!feature.contains("geometry") || !feature.contains("properties")) continue;
        const auto &geom = feature["geometry"];
        if (geom["type"] != "LineString") continue;
        const auto &props = feature["properties"];
        // ---- Street ID
        int eid = 0;
        if (props.contains("NO")) {
            eid = props["NO"].get<int>();
        } else if (props.contains("osmid")) {
            eid = props["osmid"].get<int>();
        } else continue;
        // ---- Street coordinates
        const auto &coords = geom["coordinates"];
        if (coords.size() < 2) continue;
        // ---- create segments
        for (size_t i=0; i+1<coords.size(); ++i) {
            const auto &p0 = coords[i];
            const auto &p1 = coords[i + 1];
            double x0 = p0[0];
            double y0 = p0[1];
            double z0 = (p0.size() > 2) ? p0[2].get<double>() : 0.0;
            double x1 = p1[0];
            double y1 = p1[1];
            double z1 = (p1.size() > 2) ? p1[2].get<double>() : 0.0;
            vec3d v0(x0, y0, z0);
            vec3d v1(x1, y1, z1);
            vec3d street_center = 0.5 * (v0 + v1);
            Street_Classification C = Street_Classification::street;

            Street s(eid, v0, v1, street_center, C);
            s.translate(-scene_center);
            if (!s.is_inside(boundary)) {
                // std::cout << "load_streets_GeoJSON - WARNING: street segment outside the boundary, discarded: "
                // << eid << std::endl;
                ++count;
                continue;
            }
            streets.push_back(s);
        }
    }
    if (count > 0) {
        std::cout << "load_streets_GeoJSON - WARNING: " << count
                  << " street segments discarded because outside the boundary." << std::endl;
    }
    WITH_STREETS = !streets.empty();
}

// --------------------------------------------------------------------------------------------

void City::write_streets_data(const string &filepath)
{
    vec3d c = scene_center - city_mesh.centroid();
    for (Street &s : streets) {
        s.translate(c);
    }

    nlohmann::json geojson;
    geojson["type"] = "FeatureCollection";
    geojson["name"] = "Street_segments";
    // ---- CRS (EPSG:25833)
    geojson["crs"] = {
        {"type", "name"},
        {"properties", {{"name", "urn:ogc:def:crs:EPSG::25833"}}}
    };
    geojson["features"] = nlohmann::json::array();
    for (auto &s : streets) {
        int sid  = s.get_ID();
        vec3d v0 = s.get_v0();
        vec3d v1 = s.get_v1();
        nlohmann::json feature;
        feature["type"] = "Feature";
        // ---- properties
        feature["properties"] = {
            {"NO", sid}
        };
        // ---- geometry
        nlohmann::json coords = nlohmann::json::array();
        coords.push_back({v0.x(), v0.y(), v0.z()});
        coords.push_back({v1.x(), v1.y(), v1.z()});
        feature["geometry"] = {
            {"type", "LineString"},
            {"coordinates", coords}
        };
        geojson["features"].push_back(feature);
    }
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open output GeoJSON file");
    }
    file << geojson.dump(2); // pretty print
}

// --------------------------------------------------------------------------------------------

// void City::visualize_streets(DrawableSegmentSoup &soup) const
// {
//     for (const Street &s : streets) {
//         soup.push_seg(s.get_v0(), s.get_v1(), Color::BLUE());
//     }
//     soup.thickness = 1.0;
//     soup.no_depth_test = true;
//     soup.use_gl_lines = true;
// }

}

#endif // CITY
