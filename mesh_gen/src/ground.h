#ifndef GROUND
#define GROUND

#include <cinolib/meshes/meshes.h>

#include <auxiliary.h>
#include <triangulate_with_holes.h>
#include <street.h>
#include <building.h>

namespace cinolib
{

using std::vector;
using std::string;

class Ground
{
private:
    Trimesh<>        ground_mesh;
    vector<vec3d>    holes;
    Polygonmesh<>    boundary;
    const vector<Building> *buildings = nullptr;
    const vector<Street>   *streets   = nullptr;
    bool             WITH_STREETS = false;
    vec3d            scene_center;

    void translate_back();
    void mark_streets(const Polygonmesh<> &m);

public:
    Ground()  {}
    ~Ground() {}

    void setup(const Polygonmesh<>    &boundary_mesh,
               const vector<Building> &buildings_data,
               const vector<Street>   &street_data,
               const vec3d            &center);

    void compute();

    void verify_mesh();

    void save(const string &filename);

    Trimesh<> get_ground()            const { return ground_mesh; }
    const Trimesh<> &get_ground_ref() const { return ground_mesh; }
    const Trimesh<> *get_ground_ptr() const { return &ground_mesh; }
};

// --------------------------------------------------------------------------------------------

void Ground::setup(const Polygonmesh<>    &boundary_mesh,
                   const vector<Building> &buildings_data,
                   const vector<Street>   &street_data,
                   const vec3d            &center)
{
    boundary     = boundary_mesh;
    buildings    = &buildings_data;
    streets      = &street_data;
    WITH_STREETS = !street_data.empty();
    scene_center = center;
}

// --------------------------------------------------------------------------------------------

void Ground::compute()
{
    // Start from the boundary and project it to the plane z=0.  Ground
    // elevations are kept in a hash map keyed by the exact XY position.
    Polygonmesh<> m = boundary;
    std::unordered_map<Vec2Key, double, Vec2KeyHash> z_map;
    z_map.reserve(m.num_verts() * 2 + 1024);
    Vec2KeyMap vertex_map;
    vertex_map.reserve(m.num_verts() * 2 + 1024);

    for (uint vid=0; vid<m.num_verts(); ++vid) {
        vec3d p = m.vert(vid);
        Vec2Key key{p.x(), p.y()};
        z_map.emplace(key, p.z());
        vertex_map.emplace(key, vid);
        m.vert(vid).z() = 0.;
    }
    m.update_bbox();

    auto add_vertex = [&](const vec3d &p) -> uint
    {
        Vec2Key key{p.x(), p.y()};
        auto it = vertex_map.find(key);
        if (it != vertex_map.end()) return it->second;
        uint vid = m.vert_add(vec3d(p.x(), p.y(), 0.));
        vertex_map.emplace(key, vid);
        z_map.emplace(key, p.z());
        return vid;
    };

    // Add the footprints to the mesh, reusing vertices shared by adjacent
    // buildings.  This replaces the previous octree merge, which rebuilt
    // the whole spatial index for each footprint.
    vector<vec3d> holes;
    int count = 0;
    uint duplicate_footprint_polys = 0;
    for (const Building &B : *buildings) {
        std::cout << "\r\033[K"; // deletes the line
        std::cout << "Ground::compute_ground_mesh - building " << count << " / "
                  << buildings->size() << std::flush;

        const Polygonmesh<> &fp = B.get_footprint_ref();
        std::vector<uint> remap(fp.num_verts());
        for (uint vid=0; vid<fp.num_verts(); ++vid) {
            remap[vid] = add_vertex(fp.vert(vid));
        }
        for (uint pid=0; pid<fp.num_polys(); ++pid) {
            std::vector<uint> p = fp.poly_verts_id(pid);
            for (uint &vid : p) vid = remap[vid];
            uint before = m.num_polys();
            uint added  = m.poly_add(p);
            if (added < before) duplicate_footprint_polys++;
        }
        holes.push_back(B.get_point_in_footprint());
        count++;
    }
    std::cout << std::endl;
    if (duplicate_footprint_polys > 0) {
        std::cout << "  Ground::compute - duplicate footprint polys skipped: "
                  << duplicate_footprint_polys << std::endl;
    }

    // Add streets to the mesh and assign a label to the corresponding edges.
    if (WITH_STREETS) {
        count = 0;
        for (const Street &s : *streets) {
            std::cout << "\r\033[K"; // deletes the line
            std::cout << "Ground::compute - street " << count << " / "
                      << streets->size() << std::flush;
            vec3d p0 = s.get_v0();
            vec3d p1 = s.get_v1();
            uint vid0 = add_vertex(p0);
            uint vid1 = add_vertex(p1);
            if (vid0 != vid1) {
                int eid = m.edge_id(vid0, vid1);
                if (eid == -1) eid = m.edge_add(vid0, vid1);
                m.edge_data(eid).label = s.get_ID();
            }
            count++;
        }
        std::cout << std::endl;
    }

    // triangulate the mesh with holes
    ground_mesh = triangulate_with_holes(m, holes);

    // copy edge labels
    if (WITH_STREETS) {
        mark_streets(m);
    }
    if (m.num_verts() != ground_mesh.num_verts()) {
        std::cout << "  Ground::compute - WARNING: triangulation added "
                  << ground_mesh.num_verts() - m.num_verts()
                  << " new vertices in the ground mesh!" << std::endl;
    }
    // Restore elevations by exact XY position.  Looking up by position also
    // makes this safe if Triangle reorders the input vertices.
    uint missing = 0;
    for (uint vid=0; vid<ground_mesh.num_verts(); ++vid) {
        vec3d p = ground_mesh.vert(vid);
        auto it = z_map.find(Vec2Key{p.x(), p.y()});
        if (it != z_map.end()) {
            ground_mesh.vert(vid).z() = it->second;
        } else {
            ++missing;
        }
    }
    if (missing > 0) {
        std::cout << "  Ground::compute - WARNING: " << missing
                  << " triangulated vertices have no elevation." << std::endl;
    }
    ground_mesh.update_bbox();
    ground_mesh.mesh_data().filename = m.mesh_data().filename;
}

// --------------------------------------------------------------------------------------------

void Ground::verify_mesh()
{
    double TOLL_1D = 1e-6;
    double TOLL_2D = 1e-12;

    uint non_manifold_verts = 0;
    for (uint vid=0; vid<ground_mesh.num_verts(); ++vid) {
        if (!ground_mesh.vert_is_manifold(vid)) {
            non_manifold_verts++;
            std::cout << "  non-manifold vertex ID: " << vid << " at position " << ground_mesh.vert(vid) << std::endl;
        }
    }

    uint non_manifold_edges = 0;
    for (uint eid=0; eid<ground_mesh.num_edges(); ++eid) {
        if (!ground_mesh.edge_is_manifold(eid)) {
            non_manifold_edges++;
        }
    }

    uint small_edges = 0;
    for (uint eid=0; eid<ground_mesh.num_edges(); ++eid) {
        if (ground_mesh.edge_length(eid) < TOLL_1D) {
            small_edges++;
        }
    }

    uint small_polys = 0;
    for (uint pid=0; pid<ground_mesh.num_polys(); ++pid) {
        if (ground_mesh.poly_area(pid) < TOLL_2D) {
            small_polys++;
        }
    }

    std::vector<vec3d> verts = ground_mesh.vector_verts();
    REMOVE_DUPLICATES_FROM_VEC(verts);
    uint duplicate_verts = ground_mesh.num_verts() - verts.size();

    std::vector<std::vector<uint>> polys = ground_mesh.vector_polys();
    REMOVE_DUPLICATES_FROM_VEC(polys);
    uint duplicate_polys = ground_mesh.num_polys() - polys.size();

    std::string message = "Ground mesh verification: [";
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
    message += "\n]\n";
    std::cout << message << std::flush;
}

// --------------------------------------------------------------------------------------------

void Ground::translate_back()
{
    vec3d c = scene_center /*- ground_mesh.centroid()*/;
    ground_mesh.translate(c);
}

// --------------------------------------------------------------------------------------------

void Ground::mark_streets(const Polygonmesh<> &m)
{
    if (!WITH_STREETS) return;

    // mark the edges corresponding to streets
    int count = 0;
    for (uint eid=0; eid<m.num_edges(); ++eid) {
        auto vids = m.edge_vert_ids(eid);
        int eid_tri = ground_mesh.edge_id(vids);
        if (eid_tri == -1) {
            count++;
            continue;
        }
        ground_mesh.edge_data(eid_tri).label = m.edge_data(eid).label;

        // if the edge corresponds to a street, set the z-coordinate of the vertices
        // perchè la z delle strade è sbagliata?
        if (ground_mesh.edge_data(eid_tri).label != -1) {
            for (uint vid : ground_mesh.edge_vert_ids(eid_tri)) {
                double avg_z = 0.;
                for (uint nbr : ground_mesh.adj_v2v(vid)) {
                    avg_z += ground_mesh.vert(nbr).z();
                }
                avg_z /= ground_mesh.adj_v2v(vid).size();
                ground_mesh.vert(vid).z() = avg_z;
            }
        }
    }
    if (count > 0) {
        std::cout << "  Ground::mark_streets - WARNING: triangulation modified " << count
                  << " edges in the ground mesh!" << std::endl;
    }
}

// --------------------------------------------------------------------------------------------

void Ground::save(const string &filename)
{
    translate_back();
    ground_mesh.save(filename.c_str());
}

}

#endif // GROUND
