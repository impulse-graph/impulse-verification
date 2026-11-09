package org.impulsegraph.builder;

import org.impulsegraph.builder.api.*;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SnapshotBuilderCliValidationTest {

	@Test
	void testCliToolingInspectionAndNormativeValidation() throws Exception {
		Path tempSnapshot = Files.createTempFile("impulse_cli_test_", ".imps");

		try {
			// Build a 3-domain snapshot
			DomainDefinition userDomain = DomainDefinition.builder(100L).idWidth(PrimitiveWidth.UINT32).build();

			DomainDefinition groupDomain = DomainDefinition.builder(50L).idWidth(PrimitiveWidth.UINT32).build();

			DomainDefinition roleDomain = DomainDefinition.builder(20L).idWidth(PrimitiveWidth.UINT32).build();

			List<SimpleEdgeSource.Edge> userGroupEdges = List.of(new SimpleEdgeSource.Edge(0, 1),
					new SimpleEdgeSource.Edge(0, 2), new SimpleEdgeSource.Edge(1, 2), new SimpleEdgeSource.Edge(10, 20),
					new SimpleEdgeSource.Edge(99, 49));

			List<SimpleEdgeSource.Edge> groupRoleEdges = List.of(new SimpleEdgeSource.Edge(1, 0),
					new SimpleEdgeSource.Edge(2, 5), new SimpleEdgeSource.Edge(20, 15));

			RelationDefinition userToGroup = RelationDefinition.builder("User", "Group")
					.addTopology(Topology.CSR, CompressionScheme.RAW).addTopology(Topology.CSC, CompressionScheme.RAW)
					.dataSource(new SimpleEdgeSource(userGroupEdges)).build();

			RelationDefinition groupToRole = RelationDefinition.builder("Group", "Role")
					.addTopology(Topology.CSR, CompressionScheme.RAW).dataSource(new SimpleEdgeSource(groupRoleEdges))
					.build();

			SnapshotBuilder.create().addDomain("User", userDomain).addDomain("Group", groupDomain)
					.addDomain("Role", roleDomain).addRelation("USER_TO_GROUP", userToGroup)
					.addRelation("GROUP_TO_ROLE", groupToRole).addFooterMetadata("environment", "ci_validation")
					.addFooterMetadata("builder", "impulse-builder-java").writeTo(tempSnapshot);

			assertTrue(Files.exists(tempSnapshot));
			assertTrue(Files.size(tempSnapshot) > 4096);

			// Find impulse CLI binary
			String[] candidatePaths = {System.getProperty("impulse.cli.path", ""),
					System.getenv().getOrDefault("IMPULSE_CLI_PATH", ""),
					"/Users/jesse/impulse/impulse-graph-tooling/target/release/impulse-graph",
					"/Users/jesse/impulse/impulse-graph-tooling/target/debug/impulse-graph"};

			File cliBinary = null;
			for (String candidate : candidatePaths) {
				if (!candidate.isBlank()) {
					File f = new File(candidate);
					if (f.exists() && f.canExecute()) {
						cliBinary = f;
						break;
					}
				}
			}

			if (cliBinary == null) {
				System.out.println(
						"Note: Native impulse-graph CLI binary not found on candidate paths. Skipping CLI verification.");
				return;
			}

			System.out.println("Executing CLI validation using: " + cliBinary.getAbsolutePath());

			// 1. Run impulse-graph inspect
			ProcessBuilder inspectPb = new ProcessBuilder(cliBinary.getAbsolutePath(), "inspect",
					tempSnapshot.toAbsolutePath().toString());
			inspectPb.redirectErrorStream(true);
			Process inspectProc = inspectPb.start();

			StringBuilder inspectOutput = new StringBuilder();
			try (BufferedReader reader = new BufferedReader(new InputStreamReader(inspectProc.getInputStream()))) {
				String line;
				while ((line = reader.readLine()) != null) {
					inspectOutput.append(line).append("\n");
				}
			}
			int inspectExit = inspectProc.waitFor();
			System.out.println("--- impulse-graph inspect output ---");
			System.out.println(inspectOutput);
			assertEquals(0, inspectExit, "impulse-graph inspect must exit with code 0. Output:\n" + inspectOutput);

			// 2. Run impulse-graph snapshot validate --strict-alignment
			ProcessBuilder validatePb = new ProcessBuilder(cliBinary.getAbsolutePath(), "snapshot", "validate",
					"--strict-alignment", tempSnapshot.toAbsolutePath().toString());
			validatePb.redirectErrorStream(true);
			Process validateProc = validatePb.start();

			StringBuilder validateOutput = new StringBuilder();
			try (BufferedReader reader = new BufferedReader(new InputStreamReader(validateProc.getInputStream()))) {
				String line;
				while ((line = reader.readLine()) != null) {
					validateOutput.append(line).append("\n");
				}
			}
			int validateExit = validateProc.waitFor();
			System.out.println("--- impulse-graph snapshot validate output ---");
			System.out.println(validateOutput);
			assertEquals(0, validateExit,
					"impulse-graph snapshot validate --strict-alignment must exit with code 0. Output:\n"
							+ validateOutput);

		} finally {
			Files.deleteIfExists(tempSnapshot);
		}
	}
}
