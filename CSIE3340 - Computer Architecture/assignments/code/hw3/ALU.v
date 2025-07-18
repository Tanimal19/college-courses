module ALU (
    input  [31:0] data1_i,
    input  [31:0] data2_i,
    input  [2:0]  ALUCtrl_i,
    output reg [31:0] data_o,
    output reg zero_o
);
    always @(*) begin
        case (ALUCtrl_i)
            // addition
            3'b000: data_o = data1_i + data2_i;

            // subtraction
            3'b001: data_o = data1_i - data2_i;

            // bitwise and
            3'b010: data_o = data1_i & data2_i;

            // bitwise or
            3'b011: data_o = data1_i | data2_i;

            // bitwise xor
            3'b100: data_o = data1_i ^ data2_i;

            // left shfit
            3'b101: data_o = data1_i << (data2_i[4:0]);

            // arithmetic right shift
            3'b110: data_o = $signed(data1_i) >>> (data2_i[4:0]);

            // logical right shift
            3'b111: data_o = data1_i >> (data2_i[4:0]);
        endcase

        zero_o = (data_o == 32'b0) ? 1'b1 : 1'b0;
    end
endmodule
