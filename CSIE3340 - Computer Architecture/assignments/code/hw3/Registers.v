module Registers (
    input wire clk_i,
    input wire [4:0] RS1addr_i,
    input wire [4:0] RS2addr_i,
    input wire [4:0] RDaddr_i,
    input wire [31:0] RDdata_i,
    input wire RegWrite_i,
    output wire [31:0] RS1data_o,
    output wire [31:0] RS2data_o
);
    // register file
    reg [31:0] registers [31:0];

    initial begin
        registers[0] = 32'b0;
    end

    // read
    assign RS1data_o = registers[RS1addr_i];
    assign RS2data_o = registers[RS2addr_i];

    // write
    always @(posedge clk_i) begin
        // ignore writes to register 0
        if (RegWrite_i && RDaddr_i != 5'b00000) begin
            registers[RDaddr_i] <= RDdata_i;
        end
    end
endmodule
