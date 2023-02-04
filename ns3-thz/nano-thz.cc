#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/thz-mac-nano.h"
#include "ns3/thz-channel.h"
#include "ns3/thz-mac-nano-helper.h"
#include "ns3/thz-phy-nano-helper.h"
#include "ns3/thz-directional-antenna-helper.h"
#include "ns3/traffic-generator-helper.h"
#include "ns3/thz-energy-model-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("Nano");

int main(int argc, char* argv[])
{
  Time::SetResolution(Time::FS ); // femtoseconds
  uint8_t packetLength = 75;
  uint8_t frameLength = packetLength + 8 + 20 + 8 + 17;   // UDP - IP - LLC - MAC
  int seed_run = 1;
  RngSeedManager seed;
  seed.SetRun(seed_run);

  uint8_t numNodes = 3;
  NodeContainer nodes;
  nodes.Create(numNodes);

  THzEnergyModelHelper energy;
  energy.SetEnergyModelAttribute("THzEnergyModelInitialEnergy",StringValue("0.0"));
  energy.SetEnergyModelAttribute("DataCallbackEnergy",DoubleValue(65));
  energy.Install(nodes);

  Ptr<THzChannel> thzChan = CreateObject<THzChannel>();    // noise floor -110 dBm

  THzMacNanoHelper thzMac = THzMacNanoHelper::Default();
  thzMac.Set("FrameLength",UintegerValue(frameLength));

  bool rtsOn = 0;
  if (rtsOn)
      thzMac.Set("EnableRts",StringValue("1"));
  else
      thzMac.Set("EnableRts",StringValue("0"));

  Config::SetDefault("ns3::THzSpectrumValueFactory::NumSample", DoubleValue(9));   // num Subfreq
  
  THzPhyNanoHelper thzPhy = THzPhyNanoHelper::Default ();
  thzPhy.SetPhyAttribute("PulseDuration", TimeValue(FemtoSeconds(100)));       // Tp
  thzPhy.SetPhyAttribute("Beta", DoubleValue(100));     // beta = Ts/Tp
  thzPhy.SetPhyAttribute("TxPower", DoubleValue(-20));  // dBm

  THzDirectionalAntennaHelper thzDirAntenna = THzDirectionalAntennaHelper::Default ();
  THzHelper thz;

  NetDeviceContainer devices = thz.Install (nodes, thzChan, thzPhy, thzMac, thzDirAntenna);

  // Dat vi tri thiet bi
  MobilityHelper ue;
  ue.SetPositionAllocator("ns3::UniformDiscPositionAllocator",
                                    "X", DoubleValue (0.0),
                                    "Y", DoubleValue (0.0),
                                    "rho", DoubleValue (0.01));     // theo met
  ue.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  ue.Install(nodes);

  // Internet protocol
  InternetStackHelper internet;
  internet.Install(nodes);

  Ipv4AddressHelper ipv4;
  ipv4.SetBase("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer iface = ipv4.Assign(devices);

  // arp cache init
  Ptr<ArpCache> arp = CreateObject<ArpCache>();
  arp->SetAliveTimeout(Seconds(3600));
  for (uint16_t i = 0; i < nodes.GetN(); i++) // n node
    {
      Ptr<Ipv4L3Protocol> ip = nodes.Get(i)->GetObject<Ipv4L3Protocol>();
      NS_ASSERT(ip != 0);
      // n interface tren device
      uint32_t ninter = ip->GetNInterfaces();
      for (uint32_t j = 0; j < ninter; j++)
        {
          Ptr<Ipv4Interface> ipIface = ip->GetInterface(j);
          NS_ASSERT(ipIface != 0);
          Ptr<NetDevice> device = ipIface->GetDevice();
          NS_ASSERT(device != 0);
          Mac48Address macAddr = Mac48Address::ConvertFrom(device->GetAddress());
          for (uint32_t k = 0; k < ipIface->GetNAddresses(); k++)
            {
              Ipv4Address ipAddr = ipIface->GetAddress(k).GetLocal();
              if (ipAddr == Ipv4Address::GetLoopback())   // skip lo
                {
                  continue;
                }
              ArpCache::Entry * entry = arp->Add(ipAddr);   // them ip vao arp
              Ipv4Header ipHeader;
              Ptr<Packet> packet = Create<Packet> ();
              packet->AddHeader(ipHeader);
              
              entry->MarkWaitReply(ArpCache::Ipv4PayloadHeaderPair(packet, ipHeader));  // wait cho response routing
              entry->MarkAlive(macAddr);    // ok them mac
            }
        }
    }
  // thiet lap xong arp cache

  for (uint16_t i = 0; i < nodes.GetN(); i++)
    {
      Ptr<Ipv4L3Protocol> ip = nodes.Get(i)->GetObject<Ipv4L3Protocol>();
      NS_ASSERT (ip != 0);
      uint32_t ninter = ip->GetNInterfaces();
      for (uint32_t j = 0; j < ninter; j++)
        {
          Ptr<Ipv4Interface> ipIface = ip->GetInterface(j);
          ipIface->SetArpCache(arp);    // ip --- mac
        }
    }

  TrafficGeneratorHelper Traffic;
  Traffic.SetAttribute("Mean", DoubleValue(300));       // Tb time truyen giua 2 goi tin
  Traffic.SetAttribute("PacketSize",UintegerValue(packetLength));
  ApplicationContainer Apps = Traffic.Install(nodes);
  Apps.Start(MicroSeconds(200));
  Apps.Stop(MilliSeconds(2000));

  Simulator::Stop(MilliSeconds(100 + 0.000001));    // run timeout

  Simulator::Run();
  Simulator::Destroy();
  return 0;
}
