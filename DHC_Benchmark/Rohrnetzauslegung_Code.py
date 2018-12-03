    def size_hydronic_network(self, nodelist, network_type, dT_heating=20.0,
                              dT_cooling=6.0, dp_set=200.0):
        """Calculates diameters of hydronic network with given nodelist

        Consumption nodes in `self.uesgraphs` should have an attribute
        `'max_demand_' + network_type` (e.g. `max_demand_heating`) to calculate
        pipe diameters accordingly.

        So far, this only marks the paths of each max_demand.

        Parameters
        ----------

        nodelist : list
            List with network nodes of a network in uesgraph
        network_type : str
            {'heating', 'cooling'}
        dT_heating : float
            Design temperature difference between supply and return for a
            heating network
        dT_cooling : float
            Design temperature difference between supply and return for a
            cooling network
        dp_set : float
            Specific design pressure drop for pipe network in Pa/m
        """
        assert network_type in ['heating', 'cooling'], 'Unknown network'

        if network_type == 'heating':
            subgraph = self.uesgraph.create_subgraphs('heating')['default']
        #            save_as = os.path.join('D:\\'
        #                                    'Eclipse',
        #                                    'workspace',
        #                                    'uesgraphs',
        #                                    'workspace',
        #                                    'generator',
        #                                   '10x_Generator.png')
        #            vis = visuals.Visuals(subgraph)
        #            vis.show_network(save_as=save_as, show_plot=False,
        #                                     labels='heating',
        ##                                     show_diameters=True,
        #                                     )

        diameters = [0.015,
                     0.02,
                     0.025,
                     0.032,
                     0.04,
                     0.05,
                     0.065,
                     0.08,
                     0.1,
                     0.125,
                     0.15,
                     0.2,
                     0.25,
                     0.3,
                     0.35,
                     0.4,
                     0.45,
                     0.5,
                     0.6,
                     0.7,
                     0.8,
                     0.9,
                     1.0,
                     1.1,
                     1.2]

        # Sum up maximum demands per edge
        supplies = []
        for node in nodelist:
            if self.uesgraph.node[node]['is_supply_' + network_type] is True:
                supplies.append(node)
        supply = supplies[0]  # Currently only 1 supported supply per network
        edgelist = []
        for node in nodelist:
            if self.uesgraph.node[node]['is_supply_' + network_type] is False:
                if 'max_demand_' + network_type in self.uesgraph.node[node]:
                    path = nx.shortest_path(subgraph, supply, node)
                    max_demand = self.uesgraph.node[node][
                        'max_demand_' + network_type]
                    for i in range(len(path) - 1):
                        node_0 = path[i]
                        node_1 = path[i + 1]
                        if 'max_demand_' + network_type in self.uesgraph.edge[
                            node_0][node_1]:
                            self.uesgraph.edge[node_0][node_1][
                                'max_demand_' + network_type] += max_demand
                        else:
                            self.uesgraph.edge[node_0][node_1][
                                'max_demand_' + network_type] = max_demand
                        edgelist.append([node_0, node_1])

        # Calculate nominal mass flow rates according to maximum demand
        if network_type == 'heating':
            dT = dT_heating
        elif network_type == 'cooling':
            dT = dT_cooling
        for edge in edgelist:
            max_demand = self.uesgraph.edge[edge[0]][edge[1]]['max_demand_' +
                                                              network_type]
            cp = 4180  # J/(kg*K)
            m_flow = max_demand / (cp * dT)
            self.uesgraph.edge[edge[0]][edge[1]][
                'm_flow_' + network_type] = m_flow

        # Calculate pressure drops and assign diameter
        for edge in edgelist:
            m_flow = self.uesgraph.edge[edge[0]][edge[1]][
                'm_flow_' + network_type]
            dp_spec = 1e99  # Pa/m
            i = 0
            while dp_spec > dp_set:
                diameter = diameters[i]
                dp_spec = 8 * 0.025 / (diameter ** 5 *
                                       math.pi ** 2 *
                                       983) * m_flow ** 2
                i += 1
            self.uesgraph.edge[edge[0]][edge[1]]['diameter'] = diameter