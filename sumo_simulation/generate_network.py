import os
import sys
import subprocess
import random
import xml.etree.ElementTree as ET

def generate_sumo_files(output_dir="sumo_simulation", num_vehicles=30, simulation_steps=300):
    """
    Generates a SUMO network grid (.net.xml), route definitions (.rou.xml), 
    and main config (.sumocfg) for IoV simulation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    nodes_file = os.path.join(output_dir, "grid.nod.xml")
    edges_file = os.path.join(output_dir, "grid.edg.xml")
    net_file = os.path.join(output_dir, "grid.net.xml")
    rou_file = os.path.join(output_dir, "grid.rou.xml")
    cfg_file = os.path.join(output_dir, "grid.sumocfg")

    # 1. Create Nodes XML (3x3 grid)
    nodes_elem = ET.Element("nodes")
    grid_size = 3
    spacing = 500  # 500 meters between RSUs/Intersections
    
    for row in range(grid_size):
        for col in range(grid_size):
            node_id = f"node_{row}_{col}"
            ET.SubElement(nodes_elem, "node", {
                "id": node_id,
                "x": str(col * spacing),
                "y": str(row * spacing),
                "type": "traffic_light" if (row + col) % 2 == 0 else "priority"
            })
            
    tree = ET.ElementTree(nodes_elem)
    tree.write(nodes_file, encoding="utf-8", xml_declaration=True)

    # 2. Create Edges XML
    edges_elem = ET.Element("edges")
    edge_list = []
    
    for row in range(grid_size):
        for col in range(grid_size):
            curr_id = f"node_{row}_{col}"
            # East connection
            if col < grid_size - 1:
                right_id = f"node_{row}_{col+1}"
                e1 = f"e_{curr_id}_to_{right_id}"
                e2 = f"e_{right_id}_to_{curr_id}"
                ET.SubElement(edges_elem, "edge", {"id": e1, "from": curr_id, "to": right_id, "priority": "3", "numLanes": "2", "speed": "16.67"}) # ~60 km/h
                ET.SubElement(edges_elem, "edge", {"id": e2, "from": right_id, "to": curr_id, "priority": "3", "numLanes": "2", "speed": "16.67"})
                edge_list.extend([e1, e2])
            # North connection
            if row < grid_size - 1:
                up_id = f"node_{row+1}_{col}"
                e1 = f"e_{curr_id}_to_{up_id}"
                e2 = f"e_{up_id}_to_{curr_id}"
                ET.SubElement(edges_elem, "edge", {"id": e1, "from": curr_id, "to": up_id, "priority": "3", "numLanes": "2", "speed": "16.67"})
                ET.SubElement(edges_elem, "edge", {"id": e2, "from": up_id, "to": curr_id, "priority": "3", "numLanes": "2", "speed": "16.67"})
                edge_list.extend([e1, e2])
                
    tree = ET.ElementTree(edges_elem)
    tree.write(edges_file, encoding="utf-8", xml_declaration=True)

    # 3. Use netconvert if available, or fallback to generating structured network
    try:
        cmd = ["netconvert", "--node-files", nodes_file, "--edge-files", edges_file, "--output-file", net_file]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("[+] Generated SUMO Network via netconvert successfully.")
    except Exception as e:
        print(f"[-] netconvert binary not in path, using raw xml network definition.")
        # Save raw net file
        net_file = edges_file

    # 4. Create Vehicle Routes XML
    routes_elem = ET.Element("routes")
    # Vehicle Type (Car)
    ET.SubElement(routes_elem, "vType", {
        "id": "car",
        "accel": "2.6",
        "decel": "4.5",
        "sigma": "0.5",
        "length": "4.5",
        "maxSpeed": "22.22"  # ~80 km/h
    })

    # Create defined multi-edge routes
    routes_dict = {
        "route_horizontal_1": ["e_node_0_0_to_node_0_1", "e_node_0_1_to_node_0_2"],
        "route_horizontal_2": ["e_node_1_0_to_node_1_1", "e_node_1_1_to_node_1_2"],
        "route_vertical_1": ["e_node_0_0_to_node_1_0", "e_node_1_0_to_node_2_0"],
        "route_vertical_2": ["e_node_0_1_to_node_1_1", "e_node_1_1_to_node_2_1"],
        "route_zigzag": ["e_node_0_0_to_node_0_1", "e_node_0_1_to_node_1_1", "e_node_1_1_to_node_1_2", "e_node_1_2_to_node_2_2"]
    }

    for r_id, r_edges in routes_dict.items():
        ET.SubElement(routes_elem, "route", {"id": r_id, "edges": " ".join(r_edges)})

    route_keys = list(routes_dict.keys())
    for i in range(num_vehicles):
        depart_time = random.uniform(0, 20)
        route_choice = random.choice(route_keys)
        ET.SubElement(routes_elem, "vehicle", {
            "id": f"veh_{i}",
            "type": "car",
            "route": route_choice,
            "depart": f"{depart_time:.1f}"
        })

    tree = ET.ElementTree(routes_elem)
    tree.write(rou_file, encoding="utf-8", xml_declaration=True)

    # 5. Create Config XML
    cfg_elem = ET.Element("configuration")
    input_elem = ET.SubElement(cfg_elem, "input")
    ET.SubElement(input_elem, "net-file", {"value": "grid.net.xml"})
    ET.SubElement(input_elem, "route-files", {"value": "grid.rou.xml"})
    
    time_elem = ET.SubElement(cfg_elem, "time")
    ET.SubElement(time_elem, "begin", {"value": "0"})
    ET.SubElement(time_elem, "end", {"value": str(simulation_steps)})

    tree = ET.ElementTree(cfg_elem)
    tree.write(cfg_file, encoding="utf-8", xml_declaration=True)
    print(f"[+] SUMO simulation environment created in '{output_dir}'.")

if __name__ == "__main__":
    generate_sumo_files()
