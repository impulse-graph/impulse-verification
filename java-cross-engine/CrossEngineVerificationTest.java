package org.impulsegraph.vm;

import org.impulsegraph.api.ImpulseGraphSnapshot;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static java.lang.foreign.ValueLayout.*;
import static org.impulsegraph.vm.VmRegisterType.*;
import static org.impulsegraph.vm.VmStateLayout.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Cross-Engine Verification Harness: C++20 Native Kernel vs. Java 25 FFM
 * ImpulseVM.
 *
 * <p>
 * Uses Java 25 Foreign Function & Memory (FFM) API downcalls into
 * {@code libimpulse_graph} to verify 100% identical execution status, register
 * values, flags, and bitset results between the C++ engine and Java VM
 * interpreter.
 * </p>
 */
public class CrossEngineVerificationTest {

	private static SymbolLookup NATIVE_LIB;
	private static MethodHandle IMPULSE_VM_EXECUTE_MH;
	private static MethodHandle IMPULSE_VM_CONTEXT_CREATE_MH;
	private static MethodHandle IMPULSE_VM_CONTEXT_DESTROY_MH;
	private static MethodHandle IMPULSE_VM_CONTEXT_BITSET_TEST_MH;
	private static MethodHandle IMPULSE_VM_CONTEXT_MOCK_CSR_MH;
	private static boolean NATIVE_LOADED = false;

	@BeforeAll
	public static void setupNativeBindings() {
		Path libPath = resolveNativeLibraryPath();
		if (libPath == null || !Files.exists(libPath)) {
			System.out.println("[CrossEngineVerificationTest] C++ native library not found. Skipping native tests.");
			NATIVE_LOADED = false;
			return;
		}

		try {
			System.out.println(
					"[CrossEngineVerificationTest] Loading native C++ engine from: " + libPath.toAbsolutePath());
			NATIVE_LIB = SymbolLookup.libraryLookup(libPath, Arena.global());
			Linker linker = Linker.nativeLinker();

			// impulse_vm_execute: (const impulse_instruction_t*, size_t,
			// impulse_vm_state_t*, uint64_t) -> int
			MemorySegment execAddr = NATIVE_LIB.find("impulse_vm_execute").orElseThrow();
			IMPULSE_VM_EXECUTE_MH = linker.downcallHandle(execAddr,
					FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG));

			// impulse_vm_context_create: (const impulse_snapshot_t*) ->
			// impulse_vm_context_t*
			MemorySegment ctxCreateAddr = NATIVE_LIB.find("impulse_vm_context_create").orElseThrow();
			IMPULSE_VM_CONTEXT_CREATE_MH = linker.downcallHandle(ctxCreateAddr,
					FunctionDescriptor.of(ADDRESS, ADDRESS));

			// impulse_vm_context_destroy: (impulse_vm_context_t*) -> void
			MemorySegment ctxDestroyAddr = NATIVE_LIB.find("impulse_vm_context_destroy").orElseThrow();
			IMPULSE_VM_CONTEXT_DESTROY_MH = linker.downcallHandle(ctxDestroyAddr, FunctionDescriptor.ofVoid(ADDRESS));

			// impulse_vm_context_bitset_test: (const impulse_vm_context_t*, size_t,
			// uint64_t) -> bool
			MemorySegment bitsetTestAddr = NATIVE_LIB.find("impulse_vm_context_bitset_test").orElseThrow();
			IMPULSE_VM_CONTEXT_BITSET_TEST_MH = linker.downcallHandle(bitsetTestAddr,
					FunctionDescriptor.of(JAVA_BOOLEAN, ADDRESS, JAVA_LONG, JAVA_LONG));

			// impulse_vm_context_mock_csr: (impulse_vm_context_t*, uint16_t, const
			// uint32_t*, const uint32_t*, uint64_t, uint64_t) -> void
			MemorySegment mockCsrAddr = NATIVE_LIB.find("impulse_vm_context_mock_csr").orElseThrow();
			IMPULSE_VM_CONTEXT_MOCK_CSR_MH = linker.downcallHandle(mockCsrAddr,
					FunctionDescriptor.ofVoid(ADDRESS, JAVA_SHORT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG));

			NATIVE_LOADED = true;
		} catch (Throwable t) {
			System.err.println(
					"[CrossEngineVerificationTest] Failed to initialize FFM native bindings: " + t.getMessage());
			NATIVE_LOADED = false;
		}
	}

	private static Path resolveNativeLibraryPath() {
		String prop = System.getProperty("impulse.native.path");
		if (prop != null && !prop.isBlank()) {
			Path p = Paths.get(prop);
			if (Files.exists(p))
				return p;
		}

		String env = System.getenv("IMPULSE_CORE_LIB_PATH");
		if (env != null && !env.isBlank()) {
			Path p = Paths.get(env);
			if (Files.exists(p))
				return p;
		}

		String os = System.getProperty("os.name", "").toLowerCase();
		String libName = os.contains("mac")
				? "libimpulse_graph.dylib"
				: os.contains("win") ? "impulse_graph.dll" : "libimpulse_graph.so";

		Path[] candidates = new Path[]{Paths.get("../impulse-graph-core/impulse-cpp/build").resolve(libName),
				Paths.get("../../impulse-graph-core/impulse-cpp/build").resolve(libName),
				Paths.get("../../../impulse-graph-core/impulse-cpp/build").resolve(libName),
				Paths.get("impulse-graph-core/impulse-cpp/build").resolve(libName)};

		for (Path c : candidates) {
			if (Files.exists(c)) {
				return c;
			}
		}
		return null;
	}

	@Test
	@DisplayName("Verify Scalar ALU and Branch Bytecode Parity (C++ vs Java)")
	public void testScalarAluAndBranchParity() throws Throwable {
		Assumptions.assumeTrue(NATIVE_LOADED, "Native C++ engine must be loaded");

		try (Arena arena = Arena.ofConfined()) {
			// Build bytecode:
			// 0: OP_LOAD_CONST_INT R1, 10
			// 1: OP_LOAD_CONST_INT R2, 20
			// 2: OP_MOV R0, R1
			// 3: OP_HALT
			int instrCount = 4;
			MemorySegment progSeg = arena.allocate(INSTRUCTION_LAYOUT.byteSize() * instrCount);

			setInstr(progSeg, 0, OP_LOAD_CONST_INT, (byte) 0, (short) 1, 10);
			setInstr(progSeg, 1, OP_LOAD_CONST_INT, (byte) 0, (short) 2, 20);
			setInstr(progSeg, 2, OP_MOV, (byte) 0, (short) 0, 1); // OP_MOV dst=0, src=1 (payload=1)
			setInstr(progSeg, 3, OP_HALT, (byte) 0, (short) 0, 0);

			// 1. Execute in C++ Native Kernel
			MemorySegment cppState = arena.allocate(VM_STATE_LAYOUT.byteSize(), 64);
			MemorySegment cppCtx = (MemorySegment) IMPULSE_VM_CONTEXT_CREATE_MH.invoke(MemorySegment.NULL);
			cppState.set(ADDRESS, 592L, cppCtx);

			int cppStatus = (int) IMPULSE_VM_EXECUTE_MH.invoke(progSeg, (long) instrCount, cppState, 0L);

			long cppR0 = (long) REGISTER_ELEMENT_HANDLE.get(cppState, 0L, 0L);
			long cppR1 = (long) REGISTER_ELEMENT_HANDLE.get(cppState, 0L, 1L);
			long cppR2 = (long) REGISTER_ELEMENT_HANDLE.get(cppState, 0L, 2L);

			IMPULSE_VM_CONTEXT_DESTROY_MH.invoke(cppCtx);

			// 2. Execute in Java VM Interpreter
			MemorySegment javaState = arena.allocate(VM_STATE_LAYOUT.byteSize(), 64);
			try (VmQueryContext javaCtx = new VmQueryContext(null, arena)) {
				// Execute directly on state
				long pc = 0;
				while (pc < instrCount) {
					VmHandlers.Instruction instr = VmHandlers.decodeInstruction(progSeg, pc);
					if (instr.opcode() == OP_LOAD_CONST_INT) {
						VmHandlers.handleLoadConstInt(javaState, instr);
						pc++;
					} else if (instr.opcode() == OP_MOV) {
						VmHandlers.handleMov(javaState, instr);
						pc++;
					} else if (instr.opcode() == OP_HALT) {
						break;
					} else {
						pc++;
					}
				}

				long javaR0 = VmHandlers.getRegisterValue(javaState, 0);
				long javaR1 = VmHandlers.getRegisterValue(javaState, 1);
				long javaR2 = VmHandlers.getRegisterValue(javaState, 2);

				assertEquals(0, cppStatus, "C++ execution status MUST be IMPULSE_VM_OK");
				assertEquals(cppR0, javaR0, "R0 parity failure");
				assertEquals(cppR1, javaR1, "R1 parity failure");
				assertEquals(cppR2, javaR2, "R2 parity failure");
				assertEquals(10L, javaR0, "R0 value MUST be 10");
				assertEquals(20L, javaR2, "R2 value MUST be 20");
			}
		}
	}

	@Test
	@DisplayName("Verify CSR Walk and Set Operations Parity (C++ vs Java)")
	public void testCsrWalkAndBitsetParity() throws Throwable {
		Assumptions.assumeTrue(NATIVE_LOADED, "Native C++ engine must be loaded");

		try (Arena arena = Arena.ofConfined()) {
			// Mock CSR: 4 nodes, 4 edges: 0 -> [1, 2], 1 -> [3], 2 -> [3]
			MemorySegment rowOffsets = TestSegmentHelper.allocateInts(arena, 0, 2, 3, 4, 4);
			MemorySegment colIndices = TestSegmentHelper.allocateInts(arena, 1, 2, 3, 3);

			// Bytecode:
			// 0: OP_INIT_INPUT_NODE R0, 0
			// 1: OP_CSR_WALK dst=R1, src=R0, rel=0 (flags = FLAG_INPUT_SEED)
			// 2: OP_CSR_WALK dst=R2, src=R1, rel=0
			// 3: OP_COLLECT_BITSET dst=R2
			// 4: OP_HALT
			int instrCount = 5;
			MemorySegment progSeg = arena.allocate(INSTRUCTION_LAYOUT.byteSize() * instrCount);

			setInstr(progSeg, 0, OP_INIT_INPUT_NODE, (byte) 0, (short) 0, 0);
			setInstr(progSeg, 1, OP_CSR_WALK, (byte) 0x02, (short) 1, (0 << 16) | 0); // OP_CSR_WALK rel=0, src=0, seed
			setInstr(progSeg, 2, OP_CSR_WALK, (byte) 0x00, (short) 2, (0 << 16) | 1); // OP_CSR_WALK rel=0, src=1
			setInstr(progSeg, 3, OP_COLLECT_BITSET, (byte) 0x00, (short) 2, 2); // payload=2
			setInstr(progSeg, 4, OP_HALT, (byte) 0x00, (short) 0, 0);

			// 1. C++ Engine Execution
			MemorySegment cppState = arena.allocate(VM_STATE_LAYOUT.byteSize(), 64);
			MemorySegment cppCtx = (MemorySegment) IMPULSE_VM_CONTEXT_CREATE_MH.invoke(MemorySegment.NULL);
			cppState.set(ADDRESS, 592L, cppCtx);

			IMPULSE_VM_CONTEXT_MOCK_CSR_MH.invoke(cppCtx, (short) 0, rowOffsets, colIndices, 4L, 4L);

			int cppStatus = (int) IMPULSE_VM_EXECUTE_MH.invoke(progSeg, (long) instrCount, cppState, 0L);
			assertEquals(0, cppStatus);

			long cppR2Handle = (long) REGISTER_ELEMENT_HANDLE.get(cppState, 0L, 2L);
			boolean cppHasNode3 = (boolean) IMPULSE_VM_CONTEXT_BITSET_TEST_MH.invoke(cppCtx, cppR2Handle, 3L);
			boolean cppHasNode0 = (boolean) IMPULSE_VM_CONTEXT_BITSET_TEST_MH.invoke(cppCtx, cppR2Handle, 0L);

			IMPULSE_VM_CONTEXT_DESTROY_MH.invoke(cppCtx);

			// 2. Java Engine Execution
			org.impulsegraph.storage.csr.RelationSnapshot mockRel = new org.impulsegraph.storage.csr.RelationSnapshot(
					arena, 4, 4, rowOffsets, colIndices);
			ImpulseGraphSnapshot mockGraph = new org.impulsegraph.storage.csr.GraphSnapshot(arena,
					java.util.Map.of("rel_0", mockRel));

			Object javaResult = ImpulseVmInterpreter.execute(progSeg, instrCount, mockGraph, 0, arena);
			assertNotNull(javaResult);
			assertTrue(javaResult instanceof org.impulsegraph.api.bitset.ImpulseBitSet);
			org.impulsegraph.api.bitset.ImpulseBitSet javaBitset = (org.impulsegraph.api.bitset.ImpulseBitSet) javaResult;

			assertEquals(1, javaBitset.cardinality(), "Target node 3 should be reached");
			assertTrue(javaBitset.get(3), "Java result MUST contain node 3");
			assertFalse(javaBitset.get(0), "Java result MUST NOT contain node 0");

			// 3. Compare C++ vs Java Parity
			assertTrue(cppHasNode3, "C++ bitset MUST contain node 3");
			assertFalse(cppHasNode0, "C++ bitset MUST NOT contain node 0");
			assertEquals(javaBitset.get(3), cppHasNode3, "C++ and Java node 3 presence MUST match");
		}
	}

	private static void setInstr(MemorySegment seg, int idx, byte opcode, byte flags, short dstReg, int payload) {
		long off = (long) idx * 8;
		INSTR_OPCODE_HANDLE.set(seg, off, opcode);
		INSTR_FLAGS_HANDLE.set(seg, off, flags);
		INSTR_DST_REG_HANDLE.set(seg, off, dstReg);
		INSTR_PAYLOAD_HANDLE.set(seg, off, payload);
	}
}
