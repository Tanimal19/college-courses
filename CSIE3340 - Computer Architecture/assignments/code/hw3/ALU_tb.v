`timescale 1ns/1ps

module ALU_tb();
    reg  [31:0] data1_i;
    reg  [31:0] data2_i;
    reg  [2:0]  ALUCtrl_i;
    wire [31:0] data_o;
    wire        zero_o;

    ALU alu_test(
        .data1_i(data1_i),
        .data2_i(data2_i),
        .ALUCtrl_i(ALUCtrl_i),
        .data_o(data_o),
        .zero_o(zero_o)
    );

    // test sequence
    initial begin
        $dumpfile("ALU_tb.vcd");
        $dumpvars(0, ALU_tb);

        // addition
        data1_i = 32'h0000_000A;
        data2_i = 32'h0000_0005;
        ALUCtrl_i = 3'b000;
        #1;

        // addition (overflow)
        data1_i = 32'hFFFF_FFFF;
        data2_i = 32'h0000_0001;
        ALUCtrl_i = 3'b000;
        #1;

        // subtraction
        data1_i = 32'h0000_000A;
        data2_i = 32'h0000_0005;
        ALUCtrl_i = 3'b001;
        #1;

        // subtraction (overflow)
        data1_i = 32'h0000_0005;
        data2_i = 32'h0000_0006;
        ALUCtrl_i = 3'b001;
        #1;

        // and
        data1_i = 32'hFFFF_0000;
        data2_i = 32'h0000_FFFF;
        ALUCtrl_i = 3'b010;
        #1;

        // or
        data1_i = 32'hFFFF_0000;
        data2_i = 32'h0000_FFFF;
        ALUCtrl_i = 3'b011;
        #1;

        // xor
        data1_i = 32'hFFFF_FFFF;
        data2_i = 32'hFFFF_0000;
        ALUCtrl_i = 3'b100;
        #1;

        // left shift
        data1_i = 32'h0000_0001;
        data2_i = 32'h0000_0004;
        ALUCtrl_i = 3'b101;
        #1;

        // arithmetic right shift
        data1_i = 32'h0000_0000;
        data2_i = 32'h0000_0001;
        ALUCtrl_i = 3'b110;
        #1;

        // logical right shift
        data1_i = 32'h8000_0000;
        data2_i = 32'h0000_0001;
        ALUCtrl_i = 3'b111;
        #1;

        // logical right shift (shift > 5 bits)
        data1_i = 32'h8000_0000;
        data2_i = 32'hFFFF_FF01;
        ALUCtrl_i = 3'b111;
        #1;

        $finish;
    end

endmodule