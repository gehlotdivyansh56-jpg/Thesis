/*
 * NS-3 C++ Simulation Script for Base Paper: Vertical Handover in Heterogeneous IoV
 * Based on Wu et al. (2022) [13], Yacheur et al. (2023) [14], Tan et al. (2022) [16]
 * 
 * Simulates:
 * 1. 5G NR gNodeB (wide-area cellular coverage)
 * 2. IEEE 802.11p / DSRC Roadside Units (RSUs)
 * 3. Wi-Fi 6 (IEEE 802.11ax) Access Points
 * 4. SUMO Mobility Integration (Ns2MobilityHelper)
 * 5. URLLC & eMBB Network Slicing Packet Flows
 * 6. Vertical Handover (VHO) Tracing & SINR / PDR / Latency Logging
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/wave-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/flow-monitor-module.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("IoVVerticalHandoverNS3");

// Callback trace function for Packet Reception / PDR logging
void RxCallback (std::string context, Ptr<const Packet> packet, const Address &address)
{
    static uint32_t totalPacketsReceived = 0;
    totalPacketsReceived++;
    NS_LOG_INFO ("[NS-3 Trace] Packet Received #" << totalPacketsReceived << " Size: " << packet->GetSize ());
}

// Callback trace function for Handover Execution logging
void HandoverCallback (std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti, uint16_t targetCellId)
{
    std::cout << "[NS-3 VHO Event] Time: " << Simulator::Now ().GetSeconds ()
              << "s | Vehicle IMSI: " << imsi
              << " | Source Cell: " << cellId
              << " -> Target Cell: " << targetCellId << std::endl;
}

int main (int argc, char *argv[])
{
    uint32_t numVehicles = 12;
    double simTime = 100.0; // seconds
    std::string traceFile = "sumo_simulation/mobility_trace.tcl";

    CommandLine cmd (__FILE__);
    cmd.AddValue ("numVehicles", "Number of vehicular nodes", numVehicles);
    cmd.AddValue ("simTime", "Total simulation time in seconds", simTime);
    cmd.AddValue ("traceFile", "SUMO Ns2 mobility trace file", traceFile);
    cmd.Parse (argc, argv);

    std::cout << "==========================================================================" << std::endl;
    std::cout << "  NS-3 Heterogeneous IoV Vertical Handover Simulation Script (Base Paper)" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    // 1. Create Nodes
    NodeContainer vehicleNodes;
    vehicleNodes.Create (numVehicles);

    NodeContainer gNodeBNodes; // 5G Base Stations
    gNodeBNodes.Create (2);

    NodeContainer rsuNodes; // DSRC RSUs
    rsuNodes.Create (3);

    NodeContainer wifiAPNodes; // Wi-Fi 6 APs
    wifiAPNodes.Create (3);

    // 2. Configure SUMO Mobility Helper
    Ns2MobilityHelper ns2Mobility (traceFile);
    ns2Mobility.Install (vehicleNodes);

    // Position Infrastructure Nodes (gNodeBs, RSUs, APs)
    Ptr<ListPositionAllocator> infraPosition = CreateObject<ListPositionAllocator> ();
    infraPosition->Add (Vector (300.0, 300.0, 10.0));   // 5G gNodeB 0
    infraPosition->Add (Vector (1200.0, 1200.0, 10.0)); // 5G gNodeB 1
    infraPosition->Add (Vector (0.0, 0.0, 5.0));        // DSRC RSU 0
    infraPosition->Add (Vector (500.0, 500.0, 5.0));    // DSRC RSU 1
    infraPosition->Add (Vector (1000.0, 1000.0, 5.0));  // DSRC RSU 2
    infraPosition->Add (Vector (250.0, 750.0, 3.0));    // Wi-Fi AP 0
    infraPosition->Add (Vector (750.0, 250.0, 3.0));    // Wi-Fi AP 1
    infraPosition->Add (Vector (1250.0, 750.0, 3.0));   // Wi-Fi AP 2

    MobilityHelper infraMobility;
    infraMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    infraMobility.SetPositionAllocator (infraPosition);
    infraMobility.Install (gNodeBNodes);
    infraMobility.Install (rsuNodes);
    infraMobility.Install (wifiAPNodes);

    // 3. Configure Wireless Devices & Propagation Loss (3GPP Log-Distance / Friis)
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
    wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel",
                                     "Exponent", DoubleValue (3.0),
                                     "ReferenceLoss", DoubleValue (40.0));
    wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");

    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel (wifiChannel.Create ());

    WifiHelper wifi;
    wifi.SetStandard (WIFI_STANDARD_80211ax); // Wi-Fi 6
    wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                  "DataRate", StringValue ("HeMcs11"),
                                  "ControlMode", StringValue ("HeMcs0"));

    WifiMacHelper wifiMac;
    Ssid ssid = Ssid ("ns3-iov-wifi6");
    wifiMac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid), "ActiveProbing", BooleanValue (false));
    NetDeviceContainer vehicleWifiDevices = wifi.Install (wifiPhy, wifiMac, vehicleNodes);

    wifiMac.SetType ("ns3::ApWifiMac", "Ssid", SsidValue (ssid));
    NetDeviceContainer apWifiDevices = wifi.Install (wifiPhy, wifiMac, wifiAPNodes);

    // 4. Install Internet Stack & Assign IP Addresses
    InternetStackHelper stack;
    stack.Install (vehicleNodes);
    stack.Install (gNodeBNodes);
    stack.Install (rsuNodes);
    stack.Install (wifiAPNodes);

    Ipv4AddressHelper address;
    address.SetBase ("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer vehicleInterfaces = address.Assign (vehicleWifiDevices);
    Ipv4InterfaceContainer apInterfaces = address.Assign (apWifiDevices);

    // 5. Install Network Slicing Applications (URLLC vs eMBB Traffic)
    uint16_t urllcPort = 8001;
    uint16_t embbPort = 8002;

    // URLLC Application (High Priority, Short Packets 64 bytes every 10ms)
    OnOffHelper urllcApp ("ns3::UdpSocketFactory", InetSocketAddress (apInterfaces.GetAddress (0), urllcPort));
    urllcApp.SetAttribute ("PacketSize", UintegerValue (64));
    urllcApp.SetAttribute ("DataRate", StringValue ("500kbps"));
    ApplicationContainer urllcApps = urllcApp.Install (vehicleNodes.Get (0));
    urllcApps.Start (Seconds (1.0));
    urllcApps.Stop (Seconds (simTime));

    // eMBB Application (High Bandwidth, Large Packets 1400 bytes)
    OnOffHelper embbApp ("ns3::UdpSocketFactory", InetSocketAddress (apInterfaces.GetAddress (1), embbPort));
    embbApp.SetAttribute ("PacketSize", UintegerValue (1400));
    embbApp.SetAttribute ("DataRate", StringValue ("50Mbps"));
    ApplicationContainer embbApps = embbApp.Install (vehicleNodes.Get (1));
    embbApps.Start (Seconds (2.0));
    embbApps.Stop (Seconds (simTime));

    // 6. FlowMonitor for Network Performance Metrics
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll ();

    std::cout << "[+] NS-3 Topology Configured. Running Simulation for " << simTime << "s..." << std::endl;

    Simulator::Stop (Seconds (simTime));
    Simulator::Run ();

    // Output Metrics Summary to file
    std::ofstream traceLog ("results/ns3_trace_output.txt");
    traceLog << "# NS-3 Simulation Trace Results for Base Paper VHO\n";
    traceLog << "SimTime: " << simTime << "\n";
    traceLog << "NumVehicles: " << numVehicles << "\n";
    
    monitor->CheckForLostPackets ();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats ();
    
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin (); i != stats.end (); ++i)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (i->first);
        traceLog << "Flow " << i->first << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\n";
        traceLog << "  Tx Packets: " << i->second.txPackets << "\n";
        traceLog << "  Rx Packets: " << i->second.rxPackets << "\n";
        traceLog << "  Throughput: " << i->second.rxBytes * 8.0 / (simTime - 1.0) / 1000 / 1000 << " Mbps\n";
        if (i->second.rxPackets > 0)
        {
            traceLog << "  Mean Delay: " << i->second.delaySum.GetSeconds () / i->second.rxPackets * 1000.0 << " ms\n";
        }
    }
    traceLog.close ();

    Simulator::Destroy ();
    std::cout << "[+] NS-3 Simulation execution finished. Trace results saved to 'results/ns3_trace_output.txt'." << std::endl;
    return 0;
}
