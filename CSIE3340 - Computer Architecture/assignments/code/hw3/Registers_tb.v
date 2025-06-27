`timescale 1ns/1ps

module Registers_tb();
    reg clk_i;
    reg [4:0] RS1addr_i, RS2addr_i, RDaddr_i;
    reg [31:0] RDdata_i;
    reg RegWrite_i;
    wire [31:0] RS1data_o, RS2data_o;

    Registers registers_test (
        .clk_i(clk_i),
        .RS1addr_i(RS1addr_i),
        .RS2addr_i(RS2addr_i),
        .RDaddr_i(RDaddr_i),
        .RDdata_i(RDdata_i),
        .RegWrite_i(RegWrite_i),
        .RS1data_o(RS1data_o),
        .RS2data_o(RS2data_o)
    );

    initial begin
        clk_i = 0;
        forever #5 clk_i = ~clk_i; // 10ns
    end

    task reset_signals;
        begin
            RS1addr_i = 5'b0;
            RS2addr_i = 5'b0;
            RDaddr_i = 5'b0;
            RDdata_i = 32'b0;
            RegWrite_i = 1'b0;
            #10;
        end
    endtask

    task testcase1;
        begin
            $display("\ntestcase1 started.");

            // attempt to write to register 0
            RS1addr_i = 5'b00000;
            RDaddr_i = 5'b00000;
            RDdata_i = 32'hFFFF_FFFF;
            RegWrite_i = 1'b1;
            #10;
            RegWrite_i = 1'b0;
            #10;
            $display("result: RS1data_o = %h, RS2data_o = %h", RS1data_o, RS2data_o);
        end
    endtask

    task testcase2;
        begin
            $display("\ntestcase2 started.");

            // attempt to write when RegWrite_i is low
            RS1addr_i = 5'b00001;
            RDaddr_i = 5'b00001;
            RDdata_i = 32'hFFFF_FFFF;
            RegWrite_i = 1'b0;
            #10;
            $display("result: RS1data_o = %h, RS2data_o = %h", RS1data_o, RS2data_o);
        end
    endtask

    task testcase3;
        begin
            $display("\ntestcase3 started.");

            // write to single register and read it back
            RS1addr_i = 5'b00001;
            RDaddr_i = 5'b00001;
            RDdata_i = 32'hFFFF_FFFF;
            RegWrite_i = 1'b1;
            #10;
            RegWrite_i = 1'b0;
            #10;
            $display("result: RS1data_o = %h, RS2data_o = %h", RS1data_o, RS2data_o);
        end
    endtask

    task testcase4;
        begin
            $display("\ntestcase4 started.");

            // write to multiple registers and read them back
            RS1addr_i = 5'b00010;
            RS2addr_i = 5'b00011;
            RDaddr_i = 5'b00010;
            RDdata_i = 32'hFFFF_FFFF;
            RegWrite_i = 1'b1;
            #10;
            RDaddr_i = 5'b00011;
            RDdata_i = 32'h1111_1111;
            #10;
            RegWrite_i = 1'b0;
            #10;
            $display("result: RS1data_o = %h, RS2data_o = %h", RS1data_o, RS2data_o);
        end
    endtask

    // test sequence
    initial begin
        $dumpfile("Registers_tb.vcd");
        $dumpvars(0, Registers_tb);

        reset_signals(); testcase1();
        reset_signals(); testcase2();
        reset_signals(); testcase3();
        reset_signals(); testcase4();

        $finish;
    end
endmodule